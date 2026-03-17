#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import shlex
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Optional


VERSION = "LNX1"


def wrap(text: str) -> str:
    return "\n".join(textwrap.wrap(text, width=78))


def box_title(title: str, subtitle: Optional[str] = None) -> str:
    lines = [title]
    if subtitle:
        lines.append(subtitle)
    width = max(len(line) for line in lines) + 4
    top = "+" + "-" * (width - 2) + "+"
    body = [f"| {line.ljust(width - 4)} |" for line in lines]
    return "\n".join([top, *body, top])


def prompt_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        print()
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nSession cancelled.")
        raise SystemExit(1)


class SimError(Exception):
    pass


class SimFS:
    def __init__(self, learner_name: str) -> None:
        self.learner_name = learner_name
        self.home_name = self._safe_name(learner_name)
        self.cwd = PurePosixPath(f"/home/{self.home_name}")
        self.nodes: dict[str, dict[str, str]] = {}
        self._seed()

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip())
        cleaned = "-".join(part for part in cleaned.split("-") if part)
        return cleaned or "learner"

    def _seed(self) -> None:
        self._add_dir("/")
        for path in [
            "/home",
            f"/home/{self.home_name}",
            f"/home/{self.home_name}/Documents",
            f"/home/{self.home_name}/Downloads",
            "/tmp",
            "/var",
            "/etc",
        ]:
            self._add_dir(path)
        self._add_file(
            f"/home/{self.home_name}/Documents/welcome.txt",
            "Linux rewards curiosity.\n",
        )
        self._add_file("/etc/os-release", 'NAME="Ubuntu"\nVERSION="24.04 LTS"\n')

    def _add_dir(self, path: str) -> None:
        self.nodes[path] = {"type": "dir", "content": ""}

    def _add_file(self, path: str, content: str = "") -> None:
        parent = str(PurePosixPath(path).parent)
        if parent not in self.nodes:
            self._add_dir(parent)
        self.nodes[path] = {"type": "file", "content": content}

    def exists(self, path: PurePosixPath) -> bool:
        return str(path) in self.nodes

    def is_dir(self, path: PurePosixPath) -> bool:
        return self.nodes.get(str(path), {}).get("type") == "dir"

    def is_file(self, path: PurePosixPath) -> bool:
        return self.nodes.get(str(path), {}).get("type") == "file"

    def resolve(self, raw: str) -> PurePosixPath:
        if raw == "~":
            raw = f"/home/{self.home_name}"
        elif raw.startswith("~/"):
            raw = f"/home/{self.home_name}/{raw[2:]}"
        path = PurePosixPath(raw)
        if not path.is_absolute():
            path = self.cwd / path
        parts = []
        for part in path.parts:
            if part in ("", "/"):
                continue
            if part == ".":
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        return PurePosixPath("/" + "/".join(parts))

    def prompt_path(self) -> str:
        if str(self.cwd) == f"/home/{self.home_name}":
            return "~"
        prefix = f"/home/{self.home_name}"
        current = str(self.cwd)
        if current.startswith(prefix + "/"):
            return "~" + current[len(prefix):]
        return current

    def list_dir(self, path: PurePosixPath, include_hidden: bool = False) -> list[str]:
        if not self.is_dir(path):
            raise SimError(f"ls: cannot access '{path}': No such directory")
        prefix = str(path)
        if prefix != "/":
            prefix += "/"
        names = []
        for node_path in self.nodes:
            if node_path == str(path):
                continue
            parent = str(PurePosixPath(node_path).parent)
            if parent != str(path):
                continue
            name = PurePosixPath(node_path).name
            if not include_hidden and name.startswith("."):
                continue
            names.append(name)
        return sorted(names)

    def mkdir(self, raw_paths: list[str]) -> str:
        if not raw_paths:
            raise SimError("mkdir: missing operand")
        for raw in raw_paths:
            path = self.resolve(raw)
            if self.exists(path):
                raise SimError(f"mkdir: cannot create directory '{raw}': File exists")
            parent = path.parent
            if not self.is_dir(parent):
                raise SimError(f"mkdir: cannot create directory '{raw}': No such file or directory")
            self._add_dir(str(path))
        return ""

    def cd(self, raw: Optional[str]) -> str:
        target = self.resolve(raw or f"/home/{self.home_name}")
        if not self.exists(target):
            raise SimError(f"cd: no such file or directory: {raw}")
        if not self.is_dir(target):
            raise SimError(f"cd: not a directory: {raw}")
        self.cwd = target
        return ""

    def touch(self, raw_paths: list[str]) -> str:
        if not raw_paths:
            raise SimError("touch: missing file operand")
        for raw in raw_paths:
            path = self.resolve(raw)
            parent = path.parent
            if not self.is_dir(parent):
                raise SimError(f"touch: cannot touch '{raw}': No such file or directory")
            if not self.exists(path):
                self._add_file(str(path), "")
        return ""

    def write_file(self, raw_path: str, content: str, append: bool = False) -> str:
        path = self.resolve(raw_path)
        parent = path.parent
        if not self.is_dir(parent):
            raise SimError(f"cannot write '{raw_path}': No such directory")
        current = self.nodes.get(str(path), {"type": "file", "content": ""})
        if current.get("type") == "dir":
            raise SimError(f"cannot write '{raw_path}': Is a directory")
        new_content = current["content"] + content if append else content
        self._add_file(str(path), new_content)
        return ""

    def cat(self, raw_paths: list[str]) -> str:
        if not raw_paths:
            raise SimError("cat: missing file operand")
        chunks = []
        for raw in raw_paths:
            path = self.resolve(raw)
            if not self.is_file(path):
                raise SimError(f"cat: {raw}: No such file")
            chunks.append(self.nodes[str(path)]["content"].rstrip("\n"))
        return "\n".join(chunks).rstrip("\n")

    def cp(self, src_raw: str, dst_raw: str) -> str:
        src = self.resolve(src_raw)
        dst = self.resolve(dst_raw)
        if not self.is_file(src):
            raise SimError(f"cp: cannot stat '{src_raw}': No such file")
        if self.is_dir(dst):
            dst = dst / src.name
        if not self.is_dir(dst.parent):
            raise SimError(f"cp: cannot create regular file '{dst_raw}': No such directory")
        self._add_file(str(dst), self.nodes[str(src)]["content"])
        return ""

    def mv(self, src_raw: str, dst_raw: str) -> str:
        src = self.resolve(src_raw)
        dst = self.resolve(dst_raw)
        if not self.exists(src):
            raise SimError(f"mv: cannot stat '{src_raw}': No such file or directory")
        if self.is_dir(dst):
            dst = dst / src.name
        if not self.is_dir(dst.parent):
            raise SimError(f"mv: cannot move to '{dst_raw}': No such directory")
        self.nodes[str(dst)] = self.nodes.pop(str(src))
        if self.cwd == src:
            self.cwd = dst
        if self.is_dir(dst):
            self._move_children(src, dst)
        return ""

    def _move_children(self, old_parent: PurePosixPath, new_parent: PurePosixPath) -> None:
        updates = {}
        old_prefix = str(old_parent) + "/"
        new_prefix = str(new_parent) + "/"
        for path, node in list(self.nodes.items()):
            if path.startswith(old_prefix):
                updates[new_prefix + path[len(old_prefix):]] = node
                del self.nodes[path]
        self.nodes.update(updates)

    def rm(self, raw_paths: list[str], recursive: bool = False) -> str:
        if not raw_paths:
            raise SimError("rm: missing operand")
        for raw in raw_paths:
            path = self.resolve(raw)
            if not self.exists(path):
                raise SimError(f"rm: cannot remove '{raw}': No such file or directory")
            if self.is_dir(path) and not recursive:
                raise SimError(f"rm: cannot remove '{raw}': Is a directory")
            if self.is_dir(path):
                prefix = str(path) + "/"
                for node_path in list(self.nodes.keys()):
                    if node_path.startswith(prefix):
                        del self.nodes[node_path]
            del self.nodes[str(path)]
        return ""

    def grep(self, needle: str, raw_path: str) -> str:
        path = self.resolve(raw_path)
        if not self.is_file(path):
            raise SimError(f"grep: {raw_path}: No such file")
        return "\n".join(
            line for line in self.nodes[str(path)]["content"].splitlines() if needle in line
        )

    def find(self, raw_path: Optional[str], name_filter: Optional[str]) -> str:
        base = self.resolve(raw_path or ".")
        if not self.is_dir(base):
            raise SimError(f"find: '{raw_path}': No such directory")
        results = []
        prefix = str(base)
        for node_path in sorted(self.nodes):
            if node_path == prefix or node_path.startswith(prefix + "/"):
                name = PurePosixPath(node_path).name
                if name_filter is None or name == name_filter:
                    results.append(node_path)
        return "\n".join(results)

    def read(self, raw_path: str) -> str:
        path = self.resolve(raw_path)
        if not self.is_file(path):
            raise SimError(f"{raw_path}: No such file")
        return self.nodes[str(path)]["content"]


@dataclass
class Lesson:
    title: str
    intro: str
    goal: str
    hint: str
    checker: Callable[[SimFS], bool]
    success: str
    checks: list[str]


class TutorialApp:
    def __init__(self, learner_name: str) -> None:
        self.learner_name = learner_name
        self.fs = SimFS(learner_name)
        self.run_id = secrets.token_hex(4)
        self.flags: set[str] = set()
        self.lessons = self._build_lessons()

    def _build_lessons(self) -> list[Lesson]:
        home = f"/home/{self.fs.home_name}"
        return [
            Lesson(
                "Orientation",
                f"{self.learner_name}, start by finding out where you are. Ubuntu begins with location awareness before anything else.",
                "Run the command that prints your current working directory.",
                "Use `pwd`.",
                lambda fs: False,
                "You checked your location. That is the baseline for every other shell action.",
                ["Use `pwd` once"],
            ),
            Lesson(
                "Listing",
                "Before moving, inspect what is around you. Strong shell habits start with reading the current directory.",
                "Use `ls` in your home directory so the terminal shows what is available there.",
                "Use `ls`.",
                lambda fs: False,
                "You looked before acting.",
                ["Run `ls` from your home directory"],
            ),
            Lesson(
                "Movement",
                f"Now move around deliberately. Go into `Documents`, then come back to your home directory.",
                "Finish this level while standing in your home directory after visiting `Documents`.",
                "Use `cd Documents` and then `cd ..` or `cd ~`.",
                lambda fs: str(fs.cwd) == home and fs.exists(PurePosixPath(f"{home}/Documents")),
                "Directory movement is under control.",
                ["Visit `Documents`", "Return to your home directory"],
            ),
            Lesson(
                "Workspaces",
                f"Create a practice area so you are not touching system paths. This simulated machine has a safe `/tmp` for you.",
                "Create `/tmp/linux-lab` and enter it.",
                "Try `mkdir /tmp/linux-lab` followed by `cd /tmp/linux-lab`.",
                lambda fs: str(fs.cwd) == "/tmp/linux-lab" and fs.is_dir(PurePosixPath("/tmp/linux-lab")),
                "You created a dedicated workspace and moved into it.",
                ["Create `/tmp/linux-lab`", "Enter `/tmp/linux-lab`"],
            ),
            Lesson(
                "Folders",
                "Linux work is easier when you structure it. Build a small project tree for practice.",
                "Inside `/tmp/linux-lab`, create `notes`, `projects`, `logs`, and `.secret`.",
                "You can create all four with one `mkdir` command.",
                lambda fs: all(
                    fs.is_dir(PurePosixPath(path))
                    for path in [
                        "/tmp/linux-lab/notes",
                        "/tmp/linux-lab/projects",
                        "/tmp/linux-lab/logs",
                        "/tmp/linux-lab/.secret",
                    ]
                ),
                "You created visible and hidden directories.",
                ["Create `notes`", "Create `projects`", "Create `logs`", "Create `.secret`"],
            ),
            Lesson(
                "Files",
                f"Next, create a text file and put something meaningful in it.",
                "Create `notes/commands.txt` containing the word `practice` on any line.",
                "Use `echo \"practice\" > notes/commands.txt` or append with `>>`.",
                lambda fs: fs.is_file(PurePosixPath("/tmp/linux-lab/notes/commands.txt"))
                and "practice" in fs.read("/tmp/linux-lab/notes/commands.txt"),
                "You created a file and wrote content into it.",
                ["Create `notes/commands.txt`", "Write `practice` into it"],
            ),
            Lesson(
                "Appending",
                "A lot of terminal work adds to existing files instead of replacing them. Extend your notes carefully.",
                "Append a second line containing `ls helps me explore` to `notes/commands.txt`.",
                "Use `echo \"ls helps me explore\" >> notes/commands.txt`.",
                lambda fs: fs.is_file(PurePosixPath("/tmp/linux-lab/notes/commands.txt"))
                and "ls helps me explore" in fs.read("/tmp/linux-lab/notes/commands.txt"),
                "You appended to the file without overwriting it.",
                ["Append `ls helps me explore` to `notes/commands.txt`"],
            ),
            Lesson(
                "Reading",
                f"Good shell work alternates between writing and inspecting. Confirm your file, then make a backup copy.",
                "End this level with `notes/commands.bak` present.",
                "Use `cat` to inspect, then `cp notes/commands.txt notes/commands.bak`.",
                lambda fs: fs.is_file(PurePosixPath("/tmp/linux-lab/notes/commands.bak")),
                "You inspected a file and copied it safely.",
                ["Use `cat` on `notes/commands.txt`", "Create `notes/commands.bak`"],
            ),
            Lesson(
                "Renaming",
                f"Moving and renaming are the same primitive in Linux. Use that to tidy your workspace.",
                "Rename `notes/commands.bak` to `projects/archive.txt`.",
                "Use `mv notes/commands.bak projects/archive.txt`.",
                lambda fs: fs.is_file(PurePosixPath("/tmp/linux-lab/projects/archive.txt"))
                and not fs.exists(PurePosixPath("/tmp/linux-lab/notes/commands.bak")),
                "You used `mv` to relocate and rename in one step.",
                ["Move `notes/commands.bak`", "End with `projects/archive.txt`"],
            ),
            Lesson(
                "Navigation",
                "Strong shell users can mix absolute and relative paths without getting lost.",
                "Visit `projects` using an absolute path, then move into `notes` using a relative path and end there.",
                "Try `cd /tmp/linux-lab/projects` followed by `cd ../notes`.",
                lambda fs: str(fs.cwd) == "/tmp/linux-lab/notes",
                "You navigated with both absolute and relative paths.",
                ["Visit `/tmp/linux-lab/projects` with an absolute path", "Move into `notes` with a relative path"],
            ),
            Lesson(
                "Batch Files",
                "One command can often handle several targets. Use that to prepare your log files efficiently.",
                "Create both `logs/session.log` and `logs/errors.log` in one command.",
                "Use `touch /tmp/linux-lab/logs/session.log /tmp/linux-lab/logs/errors.log` or equivalent.",
                lambda fs: fs.is_file(PurePosixPath("/tmp/linux-lab/logs/session.log"))
                and fs.is_file(PurePosixPath("/tmp/linux-lab/logs/errors.log")),
                "You created multiple files in one shot.",
                ["Create `logs/session.log`", "Create `logs/errors.log`"],
            ),
            Lesson(
                "Search",
                f"You now have enough material to search. Hidden paths matter too, so do not ignore them.",
                "Create `.secret/clue.txt` containing `token`, then prove you can find it.",
                "Use `echo \"token\" > .secret/clue.txt`, `ls -a`, and `find . -name clue.txt`.",
                lambda fs: fs.is_file(PurePosixPath("/tmp/linux-lab/.secret/clue.txt"))
                and "token" in fs.read("/tmp/linux-lab/.secret/clue.txt"),
                "You handled hidden content and file discovery.",
                ["Create `.secret/clue.txt` with `token`", "Use `ls -a`", "Use `find . -name clue.txt`"],
            ),
            Lesson(
                "Filtering",
                f"Text tools are where the command line starts to pay off.",
                "Use `grep` so the terminal shows the line containing `practice` from `notes/commands.txt`.",
                "Run `grep practice notes/commands.txt`.",
                lambda fs: False,
                "You filtered file content with `grep`.",
                ["Run `grep practice notes/commands.txt` and show the match"],
            ),
            Lesson(
                "Counting",
                "Once commands produce output, pipes let you summarize that output quickly.",
                "Use a pipeline so the simulator shows how many paths are returned by `find .`.",
                "Run `find . | wc -l` from `/tmp/linux-lab`.",
                lambda fs: False,
                "You combined commands with a pipeline to summarize output.",
                ["Run `find . | wc -l` from `/tmp/linux-lab`"],
            ),
            Lesson(
                "Cleanup",
                f"Last step: leave the lab clean. Remove `/tmp/linux-lab` completely.",
                "Delete the practice directory recursively from `/tmp` or another safe location.",
                "Change out of the directory first, then use `rm -r /tmp/linux-lab`.",
                lambda fs: not fs.exists(PurePosixPath("/tmp/linux-lab")),
                "You cleaned up after yourself like a careful Linux user.",
                ["Leave `/tmp/linux-lab`", "Remove `/tmp/linux-lab` recursively"],
            ),
        ]

    def run(self) -> None:
        self._intro()

        for index, lesson in enumerate(self.lessons, start=1):
            print()
            print(box_title(f"LEVEL {index}: {lesson.title}", self._ascii_subtitle(lesson.title)))
            print(wrap(lesson.intro))
            print(f"Goal: {lesson.goal}")
            print("Hint: hidden. Type `hint` to reveal it.")
            print("Checks: type `status` to inspect progress.")

            while True:
                line = prompt_input(self._prompt())
                output, level_complete = self.execute(line, lesson)

                if lesson.title == "Orientation" and line.strip() == "pwd":
                    level_complete = True

                if output:
                    print(output)

                if level_complete:
                    print(lesson.success)
                    if index < len(self.lessons):
                        time.sleep(0.8)
                        prompt_input("Press Enter for the next level...")
                    break

        print()
        print(wrap(f"Tutorial complete, {self.learner_name}. You worked through the beginner Linux flow in a simulated environment."))
        code = make_completion_code(self.learner_name, self.run_id, int(time.time()))
        print(f"Run ID: {self.run_id}")
        print(f"Completion hash: {code}")
        print(wrap(
            "Send the run ID and completion hash to the course owner after you finish."
        ))

    def _intro(self) -> None:
        print("Linux Command Line Tutorial")
        print("---------------------------")
        print(wrap(
            f"This is a guided terminal trainer based on Ubuntu's beginner command-line tutorial. "
            f"You are working inside a simulated shell, {self.learner_name}, so the tutorial stays separate from the host system."
        ))
        print(wrap(
            "Only a focused set of commands is available here: pwd, cd, ls, mkdir, touch, echo with > or >>, "
            "cat, cp, mv, rm -r, grep, find, and simple `| wc -l` pipelines, plus clear, help, hint, status, exit."
        ))
        print(wrap(
            "Run this on any operating system with Python 3. Each level unlocks only after you complete the required task."
        ))

    def _prompt(self) -> str:
        return f"{self.fs.home_name}@linux-tutor:{self.fs.prompt_path()}$ "

    def execute(
        self,
        line: str,
        lesson: Lesson,
    ) -> tuple[str, bool]:
        stripped = line.strip()
        if not stripped:
            return "", False
        if stripped == "help":
            return (
                "Available commands: pwd, cd, ls, mkdir, touch, echo >, echo >>, cat, cp, mv, rm -r, grep, find, | wc -l, clear, help, hint, status, exit",
                False,
            )
        if stripped == "hint":
            return lesson.hint, False
        if stripped == "status":
            return self._status_text(lesson), False
        if stripped == "clear":
            return "\n" * 40, False
        if stripped == "exit":
            raise SystemExit("Session ended before completion.")

        try:
            if "|" in stripped:
                output = self._run_pipeline(stripped)
            else:
                output = self._run_command(stripped)
        except SimError as exc:
            return str(exc), False

        level_complete = False
        self._record_flags(stripped, output)

        if lesson.title == "Movement":
            level_complete = "visited_documents" in self.flags and lesson.checker(self.fs)
        elif lesson.title == "Listing":
            level_complete = "used_ls_home" in self.flags
        elif lesson.title == "Appending":
            level_complete = "used_append_explore" in self.flags and lesson.checker(self.fs)
        elif lesson.title == "Reading":
            level_complete = "used_cat_commands" in self.flags and lesson.checker(self.fs)
        elif lesson.title == "Navigation":
            level_complete = (
                "used_abs_projects" in self.flags
                and "used_rel_notes" in self.flags
                and lesson.checker(self.fs)
            )
        elif lesson.title == "Batch Files":
            level_complete = "used_touch_logs" in self.flags and lesson.checker(self.fs)
        elif lesson.title == "Search":
            level_complete = (
                "used_ls_a" in self.flags
                and "used_find_clue" in self.flags
                and lesson.checker(self.fs)
            )
        elif lesson.title == "Filtering":
            level_complete = "used_grep_practice" in self.flags and "practice" in output.splitlines()
        elif lesson.title == "Counting":
            level_complete = "used_find_wc" in self.flags
        elif lesson.title != "Orientation":
            level_complete = lesson.checker(self.fs)
        return output, level_complete

    def _record_flags(self, command: str, output: str) -> None:
        normalized = command.strip()
        if normalized == "pwd":
            self.flags.add("used_pwd")
        if normalized == "ls" and str(self.fs.cwd) == f"/home/{self.fs.home_name}" and "Documents" in output:
            self.flags.add("used_ls_home")
        if normalized in {"cd Documents", "cd ./Documents"} and str(self.fs.cwd).endswith("/Documents"):
            self.flags.add("visited_documents")
        if normalized == 'echo "ls helps me explore" >> notes/commands.txt' or (
            normalized.endswith(">> notes/commands.txt") and "ls helps me explore" in normalized
        ):
            self.flags.add("used_append_explore")
        if normalized.startswith("cat ") and "commands.txt" in normalized:
            self.flags.add("used_cat_commands")
        if normalized == "cd /tmp/linux-lab/projects" and str(self.fs.cwd) == "/tmp/linux-lab/projects":
            self.flags.add("used_abs_projects")
        if normalized == "cd ../notes" and str(self.fs.cwd) == "/tmp/linux-lab/notes":
            self.flags.add("used_rel_notes")
        if normalized.startswith("touch ") and "logs/session.log" in normalized and "logs/errors.log" in normalized:
            self.flags.add("used_touch_logs")
        if normalized == "ls -a" or normalized.startswith("ls -a "):
            self.flags.add("used_ls_a")
        if normalized == "find . -name clue.txt":
            self.flags.add("used_find_clue")
        if normalized.startswith("grep practice ") and "practice" in output.splitlines():
            self.flags.add("used_grep_practice")
        if normalized == "find . | wc -l" and str(self.fs.cwd) == "/tmp/linux-lab" and output.strip().isdigit():
            self.flags.add("used_find_wc")

    def _ascii_subtitle(self, title: str) -> str:
        subtitles = {
            "Orientation": "Know where you stand",
            "Listing": "Read the room",
            "Movement": "Move with intent",
            "Workspaces": "Build a safe lab",
            "Folders": "Shape the tree",
            "Files": "Write something down",
            "Appending": "Add without erasing",
            "Reading": "Inspect before acting",
            "Renaming": "Move and rename",
            "Navigation": "Switch path styles",
            "Batch Files": "Act on many targets",
            "Search": "Find what is hidden",
            "Filtering": "Pull signal from text",
            "Counting": "Summarize with pipes",
            "Cleanup": "Leave no mess",
        }
        return subtitles.get(title, "")

    def _status_text(self, lesson: Lesson) -> str:
        checks = self._checks_for_lesson(lesson)
        lines = [box_title(f"STATUS: {lesson.title}"), f"Location: {self.fs.cwd}", "Progress:"]
        for label, done in checks:
            marker = "[x]" if done else "[ ]"
            lines.append(f"{marker} {label}")
        return "\n".join(lines)

    def _checks_for_lesson(self, lesson: Lesson) -> list[tuple[str, bool]]:
        fs = self.fs
        title = lesson.title
        if title == "Orientation":
            return [(lesson.checks[0], "used_pwd" in self.flags)]
        if title == "Listing":
            return [(lesson.checks[0], "used_ls_home" in self.flags)]
        if title == "Movement":
            return [
                (lesson.checks[0], "visited_documents" in self.flags),
                (lesson.checks[1], str(fs.cwd) == f"/home/{fs.home_name}"),
            ]
        if title == "Workspaces":
            return [
                (lesson.checks[0], fs.is_dir(PurePosixPath("/tmp/linux-lab"))),
                (lesson.checks[1], str(fs.cwd) == "/tmp/linux-lab"),
            ]
        if title == "Folders":
            return [
                (lesson.checks[0], fs.is_dir(PurePosixPath("/tmp/linux-lab/notes"))),
                (lesson.checks[1], fs.is_dir(PurePosixPath("/tmp/linux-lab/projects"))),
                (lesson.checks[2], fs.is_dir(PurePosixPath("/tmp/linux-lab/logs"))),
                (lesson.checks[3], fs.is_dir(PurePosixPath("/tmp/linux-lab/.secret"))),
            ]
        if title == "Files":
            return [
                (lesson.checks[0], fs.is_file(PurePosixPath("/tmp/linux-lab/notes/commands.txt"))),
                (lesson.checks[1], fs.is_file(PurePosixPath("/tmp/linux-lab/notes/commands.txt")) and "practice" in fs.read("/tmp/linux-lab/notes/commands.txt")),
            ]
        if title == "Appending":
            return [
                (lesson.checks[0], fs.is_file(PurePosixPath("/tmp/linux-lab/notes/commands.txt")) and "ls helps me explore" in fs.read("/tmp/linux-lab/notes/commands.txt")),
            ]
        if title == "Reading":
            return [
                (lesson.checks[0], "used_cat_commands" in self.flags),
                (lesson.checks[1], fs.is_file(PurePosixPath("/tmp/linux-lab/notes/commands.bak"))),
            ]
        if title == "Renaming":
            return [
                (lesson.checks[0], not fs.exists(PurePosixPath("/tmp/linux-lab/notes/commands.bak"))),
                (lesson.checks[1], fs.is_file(PurePosixPath("/tmp/linux-lab/projects/archive.txt"))),
            ]
        if title == "Navigation":
            return [
                (lesson.checks[0], "used_abs_projects" in self.flags),
                (lesson.checks[1], "used_rel_notes" in self.flags and str(fs.cwd) == "/tmp/linux-lab/notes"),
            ]
        if title == "Batch Files":
            return [
                (lesson.checks[0], fs.is_file(PurePosixPath("/tmp/linux-lab/logs/session.log"))),
                (lesson.checks[1], fs.is_file(PurePosixPath("/tmp/linux-lab/logs/errors.log"))),
            ]
        if title == "Search":
            return [
                (lesson.checks[0], fs.is_file(PurePosixPath("/tmp/linux-lab/.secret/clue.txt")) and "token" in fs.read("/tmp/linux-lab/.secret/clue.txt")),
                (lesson.checks[1], "used_ls_a" in self.flags),
                (lesson.checks[2], "used_find_clue" in self.flags),
            ]
        if title == "Filtering":
            return [(lesson.checks[0], "used_grep_practice" in self.flags)]
        if title == "Counting":
            return [(lesson.checks[0], "used_find_wc" in self.flags)]
        if title == "Cleanup":
            return [
                (lesson.checks[0], str(fs.cwd) != "/tmp/linux-lab"),
                (lesson.checks[1], not fs.exists(PurePosixPath("/tmp/linux-lab"))),
            ]
        return [(label, False) for label in lesson.checks]

    def _run_pipeline(self, line: str) -> str:
        left, right = [part.strip() for part in line.split("|", 1)]
        left_output = self._run_command(left)
        tokens = shlex.split(right)
        if tokens == ["wc", "-l"]:
            return str(len([chunk for chunk in left_output.splitlines() if chunk != ""]))
        raise SimError("Only simple `| wc -l` pipelines are supported here")

    def _run_command(self, line: str) -> str:
        redirect = None
        append = False
        if ">>" in line:
            command_part, target = line.split(">>", 1)
            redirect = target.strip()
            append = True
            line = command_part.strip()
        elif ">" in line:
            command_part, target = line.split(">", 1)
            redirect = target.strip()
            line = command_part.strip()

        tokens = shlex.split(line)
        if not tokens:
            return ""

        cmd, args = tokens[0], tokens[1:]
        output = ""

        if cmd == "pwd":
            output = str(self.fs.cwd)
        elif cmd == "cd":
            output = self.fs.cd(args[0] if args else None)
        elif cmd == "ls":
            include_hidden = "-a" in args
            paths = [arg for arg in args if arg != "-a"]
            target = self.fs.resolve(paths[0]) if paths else self.fs.cwd
            output = "  ".join(self.fs.list_dir(target, include_hidden=include_hidden))
        elif cmd == "mkdir":
            output = self.fs.mkdir(args)
        elif cmd == "touch":
            output = self.fs.touch(args)
        elif cmd == "echo":
            output = " ".join(args)
        elif cmd == "cat":
            output = self.fs.cat(args)
        elif cmd == "cp":
            if len(args) != 2:
                raise SimError("cp: expected source and destination")
            output = self.fs.cp(args[0], args[1])
        elif cmd == "mv":
            if len(args) != 2:
                raise SimError("mv: expected source and destination")
            output = self.fs.mv(args[0], args[1])
        elif cmd == "rm":
            recursive = "-r" in args or "-rf" in args or "-fr" in args
            targets = [arg for arg in args if not arg.startswith("-")]
            output = self.fs.rm(targets, recursive=recursive)
        elif cmd == "grep":
            if len(args) != 2:
                raise SimError("grep: expected PATTERN FILE")
            output = self.fs.grep(args[0], args[1])
        elif cmd == "find":
            search_path = None
            name_filter = None
            if args:
                search_path = args[0]
            if len(args) >= 3 and args[1] == "-name":
                name_filter = args[2]
            output = self.fs.find(search_path, name_filter)
        else:
            raise SimError(f"{cmd}: command not available in this simulator")

        if redirect is not None:
            return self.fs.write_file(redirect, output + "\n", append=append)
        return output


def make_completion_code(name: str, run_id: str, completed_at: int) -> str:
    payload = {"name": name, "run_id": run_id, "completed_at": completed_at}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("ascii").rstrip("=")
    digest = hashlib.sha256(payload_json).hexdigest()[:20]
    return f"{VERSION}.{payload_b64}.{digest}"


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] in {"-h", "--help"}:
        print("Usage:")
        print("  python tutorial_app.py")
        print("  python3 tutorial_app.py")
        return 0

    name = ""
    while not name.strip():
        name = prompt_input("Enter learner name: ").strip()
    app = TutorialApp(name)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
