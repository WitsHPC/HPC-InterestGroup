'use strict';

var obsidian = require('obsidian');

/******************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise, SuppressedError, Symbol */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    if (typeof b !== "function" && b !== null)
        throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

typeof SuppressedError === "function" ? SuppressedError : function (error, suppressed, message) {
    var e = new Error(message);
    return e.name = "SuppressedError", e.error = error, e.suppressed = suppressed, e;
};

var GeneralModal = /** @class */ (function (_super) {
    __extends(GeneralModal, _super);
    function GeneralModal(leaves, plugin) {
        var _this = _super.call(this, app) || this;
        _this.leaves = leaves;
        _this.plugin = plugin;
        return _this;
    }
    GeneralModal.prototype.open = function () {
        var _this = this;
        this.dimBackground = false;
        _super.prototype.open.call(this);
        this.chooser.setSelectedItem(1);
        this.focusTab();
        this.containerEl
            .getElementsByClassName("prompt-input-container")
            .item(0)
            .detach();
        // hotkey = this.app.hotkeyManager.bakedIds.find((e)=>e == "")
        this.scope.register(["Ctrl"], "Tab", function (e) {
            _this.chooser.setSelectedItem(_this.chooser.selectedItem + 1);
            _this.focusTab();
        });
        this.scope.register(["Ctrl", "Shift"], "Tab", function (e) {
            _this.chooser.setSelectedItem(_this.chooser.selectedItem - 1);
            _this.focusTab();
        });
        return new Promise(function (resolve) {
            _this.resolve = resolve;
        });
    };
    GeneralModal.prototype.onClose = function () {
        if (this.resolve)
            this.resolve(this.chooser.selectedItem);
    };
    GeneralModal.prototype.getSuggestions = function (query) {
        return this.leaves.map(function (leaf) { return leaf.view.getDisplayText(); });
    };
    GeneralModal.prototype.renderSuggestion = function (value, el) {
        el.setText(value);
    };
    GeneralModal.prototype.onChooseSuggestion = function (item, evt) { };
    GeneralModal.prototype.focusTab = function () {
        this.plugin.queueFocusLeaf(this.leaves[this.chooser.selectedItem]);
    };
    return GeneralModal;
}(obsidian.SuggestModal));

var CTPSettingTab = /** @class */ (function (_super) {
    __extends(CTPSettingTab, _super);
    function CTPSettingTab(plugin, settings) {
        var _this = _super.call(this, plugin.app, plugin) || this;
        _this.settings = settings;
        _this.plugin = plugin;
        return _this;
    }
    CTPSettingTab.prototype.display = function () {
        var _this = this;
        var containerEl = this.containerEl;
        containerEl.empty();
        containerEl.createEl("h2", {
            text: "Cycle through Panes Configuration",
        });
        new obsidian.Setting(containerEl)
            .setName("Only cycle through tabs with specific View Types")
            .addToggle(function (cb) {
            cb.setValue(_this.settings.useViewTypes);
            cb.onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.settings.useViewTypes = value;
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        var descEl = createFragment();
        descEl.append(createEl("p", {
            text: "If the option above is enabled: These are the View Types this Plugin will cycle through using any of the available commands.",
        }), createEl("p", {
            text: 'To add a new View Type to this List, simply run the Command: "Cycle through Panes: Enable this View Type". More advanced Users can edit and delete the Types in the text field (one per line).',
        }));
        new obsidian.Setting(containerEl)
            .setName("Enabled View Types")
            .setDesc(descEl)
            .addTextArea(function (cb) {
            var value = "";
            _this.settings.viewTypes.forEach(function (type) { return (value += type + "\n"); });
            cb.setValue(value);
            cb.setPlaceholder("markdown");
            cb.onChange(function (newValue) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            //                                                    No empty lines
                            this.settings.viewTypes = newValue
                                .split("\n")
                                .filter(function (pre) { return !!pre; });
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        new obsidian.Setting(containerEl)
            .setName("Show modal when switching tabs")
            .addToggle(function (cb) {
            cb.setValue(_this.settings.showModal);
            cb.onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.settings.showModal = value;
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        new obsidian.Setting(containerEl)
            .setName("Focus tab on release")
            .setDesc("If enabled, defer switching tabs until the ctrl key is released, similar to VS Code and Firefox")
            .addToggle(function (cb) {
            cb.setValue(_this.settings.focusLeafOnKeyUp);
            cb.onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.settings.focusLeafOnKeyUp = value;
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        new obsidian.Setting(containerEl).setName("Skip pinned tabs").addToggle(function (cb) {
            cb.setValue(_this.settings.skipPinned);
            cb.onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.settings.skipPinned = value;
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
        new obsidian.Setting(containerEl)
            .setName("Stay in current split")
            .setDesc("If enabled and the currently active file is in the sidebar, you cycle within that sidebar and can't switch to the main tabs. Use the ")
            .addToggle(function (cb) {
            cb.setValue(_this.settings.stayInSplit);
            cb.onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.settings.stayInSplit = value;
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [2 /*return*/];
                    }
                });
            }); });
        });
    };
    return CTPSettingTab;
}(obsidian.PluginSettingTab));

var DEFAULT_SETTINGS = {
    viewTypes: ["markdown", "canvas", "pdf"],
    showModal: true,
    skipPinned: false,
    stayInSplit: true,
    focusLeafOnKeyUp: false,
    useViewTypes: true,
};
var NEW_USER_SETTINGS = {
    focusLeafOnKeyUp: true,
    useViewTypes: false,
};

var CycleThroughPanes = /** @class */ (function (_super) {
    __extends(CycleThroughPanes, _super);
    function CycleThroughPanes() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.ctrlPressedTimestamp = 0;
        _this.leafIndex = 0;
        _this.keyDownFunc = _this.onKeyDown.bind(_this);
        _this.keyUpFunc = _this.onKeyUp.bind(_this);
        return _this;
    }
    CycleThroughPanes.prototype.getLeavesOfTypes = function (types) {
        var _this = this;
        var leaves = [];
        var activeLeaf = this.app.workspace.activeLeaf;
        this.app.workspace.iterateAllLeaves(function (leaf) {
            if (_this.settings.skipPinned && leaf.getViewState().pinned)
                return;
            var correctViewType = !_this.settings.useViewTypes ||
                types.contains(leaf.view.getViewType());
            if (!correctViewType)
                return;
            var isMainWindow = leaf.view.containerEl.win == window;
            var sameWindow = leaf.view.containerEl.win == activeWindow;
            var correctPane = false;
            if (isMainWindow) {
                if (_this.settings.stayInSplit) {
                    correctPane =
                        sameWindow && leaf.getRoot() == activeLeaf.getRoot();
                }
                else {
                    correctPane =
                        sameWindow &&
                            leaf.getRoot() == _this.app.workspace.rootSplit;
                }
            }
            else {
                correctPane = sameWindow;
            }
            if (correctPane) {
                leaves.push(leaf);
            }
        });
        return leaves;
    };
    CycleThroughPanes.prototype.onload = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        console.log("loading plugin: Cycle through panes");
                        return [4 /*yield*/, this.loadSettings()];
                    case 1:
                        _a.sent();
                        this.addSettingTab(new CTPSettingTab(this, this.settings));
                        this.addCommand({
                            id: "cycle-through-panes",
                            name: "Go to right tab",
                            checkCallback: function (checking) {
                                var active = _this.app.workspace.activeLeaf;
                                if (active) {
                                    if (!checking) {
                                        var leaves = _this.getLeavesOfTypes(_this.settings.viewTypes);
                                        var index = leaves.indexOf(active);
                                        if (index === leaves.length - 1) {
                                            _this.queueFocusLeaf(leaves[0]);
                                        }
                                        else {
                                            _this.queueFocusLeaf(leaves[index + 1]);
                                        }
                                    }
                                    return true;
                                }
                                return false;
                            },
                        });
                        this.addCommand({
                            id: "cycle-through-panes-reverse",
                            name: "Go to left tab",
                            checkCallback: function (checking) {
                                var active = _this.app.workspace.activeLeaf;
                                if (active) {
                                    if (!checking) {
                                        var leaves = _this.getLeavesOfTypes(_this.settings.viewTypes);
                                        var index = leaves.indexOf(active);
                                        if (index !== undefined) {
                                            if (index === 0) {
                                                _this.queueFocusLeaf(leaves[leaves.length - 1]);
                                            }
                                            else {
                                                _this.queueFocusLeaf(leaves[index - 1]);
                                            }
                                        }
                                    }
                                    return true;
                                }
                                return false;
                            },
                        });
                        this.addCommand({
                            id: "cycle-through-panes-add-view",
                            name: "Enable this View Type",
                            checkCallback: function (checking) {
                                var active = _this.app.workspace.activeLeaf;
                                if (active &&
                                    !_this.settings.viewTypes.contains(active.view.getViewType())) {
                                    if (!checking) {
                                        _this.settings.viewTypes.push(active.view.getViewType());
                                        _this.saveSettings();
                                    }
                                    return true;
                                }
                                return false;
                            },
                        });
                        this.addCommand({
                            id: "cycle-through-panes-remove-view",
                            name: "Disable this View Type",
                            checkCallback: function (checking) {
                                var active = _this.app.workspace.activeLeaf;
                                if (active &&
                                    _this.settings.viewTypes.contains(active.view.getViewType())) {
                                    if (!checking) {
                                        _this.settings.viewTypes.remove(active.view.getViewType());
                                        _this.saveSettings();
                                    }
                                    return true;
                                }
                                return false;
                            },
                        });
                        this.addCommand({
                            id: "focus-left-sidebar",
                            name: "Focus on left sidebar",
                            callback: function () {
                                app.workspace.leftSplit.expand();
                                var leaf;
                                app.workspace.iterateAllLeaves(function (e) {
                                    if (e.getRoot() == app.workspace.leftSplit) {
                                        if (e.activeTime > ((leaf === null || leaf === void 0 ? void 0 : leaf.activeTime) || 0)) {
                                            leaf = e;
                                        }
                                    }
                                });
                                _this.queueFocusLeaf(leaf);
                            },
                        });
                        this.addCommand({
                            id: "focus-right-sidebar",
                            name: "Focus on right sidebar",
                            callback: function () {
                                app.workspace.rightSplit.expand();
                                var leaf;
                                app.workspace.iterateAllLeaves(function (e) {
                                    if (e.getRoot() == app.workspace.rightSplit) {
                                        if (e.activeTime > ((leaf === null || leaf === void 0 ? void 0 : leaf.activeTime) || 0)) {
                                            leaf = e;
                                        }
                                    }
                                });
                                _this.queueFocusLeaf(leaf);
                            },
                        });
                        this.addCommand({
                            id: "focus-on-last-active-pane",
                            name: "Go to previous tab",
                            callback: function () { return __awaiter(_this, void 0, void 0, function () {
                                var leaf;
                                return __generator(this, function (_a) {
                                    this.setLeaves();
                                    this.leafIndex = (this.leafIndex + 1) % this.leaves.length;
                                    leaf = this.leaves[this.leafIndex];
                                    if (leaf) {
                                        this.queueFocusLeaf(leaf);
                                    }
                                    return [2 /*return*/];
                                });
                            }); },
                        });
                        this.addCommand({
                            id: "focus-on-last-active-pane-reverse",
                            name: "Go to next tab",
                            callback: function () { return __awaiter(_this, void 0, void 0, function () {
                                var leaf;
                                return __generator(this, function (_a) {
                                    this.setLeaves();
                                    this.leafIndex =
                                        (this.leafIndex - 1 + this.leaves.length) %
                                            this.leaves.length;
                                    leaf = this.leaves[this.leafIndex];
                                    if (leaf) {
                                        this.queueFocusLeaf(leaf);
                                    }
                                    return [2 /*return*/];
                                });
                            }); },
                        });
                        window.addEventListener("keydown", this.keyDownFunc);
                        window.addEventListener("keyup", this.keyUpFunc);
                        return [2 /*return*/];
                }
            });
        });
    };
    CycleThroughPanes.prototype.queueFocusLeaf = function (leaf) {
        if (this.settings.focusLeafOnKeyUp) {
            this.queuedFocusLeaf = leaf;
        }
        else {
            this.focusLeaf(leaf);
        }
    };
    CycleThroughPanes.prototype.focusLeaf = function (leaf) {
        if (leaf) {
            var root = leaf.getRoot();
            if (root != this.app.workspace.rootSplit && obsidian.Platform.isMobile) {
                root.openLeaf(leaf);
                leaf.activeTime = Date.now();
            }
            else {
                this.app.workspace.setActiveLeaf(leaf, { focus: true });
            }
            if (leaf.getViewState().type == "search") {
                var search = leaf.view.containerEl.find(".search-input-container input");
                search.focus();
            }
        }
    };
    CycleThroughPanes.prototype.setLeaves = function () {
        if (!this.leaves) {
            var leaves = this.getLeavesOfTypes(this.settings.viewTypes);
            leaves.sort(function (a, b) {
                return b.activeTime - a.activeTime;
            });
            this.leaves = leaves;
            this.leafIndex = leaves.indexOf(this.app.workspace.activeLeaf);
        }
    };
    CycleThroughPanes.prototype.onKeyDown = function (e) {
        if (e.key == "Control") {
            this.ctrlPressedTimestamp = e.timeStamp;
            this.ctrlKeyCode = e.code;
            // clean slate -- prevent ctrl keystroke from accidentally switching to another tab
            this.queuedFocusLeaf = undefined;
        }
    };
    CycleThroughPanes.prototype.onKeyUp = function (e) {
        var _a;
        if (e.code == this.ctrlKeyCode && this.ctrlPressedTimestamp) {
            this.ctrlPressedTimestamp = 0;
            this.leaves = null;
            (_a = this.modal) === null || _a === void 0 ? void 0 : _a.close();
            if (this.queuedFocusLeaf) {
                this.focusLeaf(this.queuedFocusLeaf);
            }
            this.modal = undefined;
        }
        if (e.code == "Tab" &&
            this.ctrlPressedTimestamp &&
            this.settings.showModal &&
            !this.modal &&
            this.leaves) {
            this.modal = new GeneralModal(this.leaves, this);
            this.modal.open();
        }
    };
    CycleThroughPanes.prototype.onunload = function () {
        console.log("unloading plugin: Cycle through panes");
        window.removeEventListener("keydown", this.keyDownFunc);
        window.removeEventListener("keyup", this.keyUpFunc);
    };
    CycleThroughPanes.prototype.loadSettings = function () {
        return __awaiter(this, void 0, void 0, function () {
            var userSettings;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.loadData()];
                    case 1:
                        userSettings = _a.sent();
                        this.settings = Object.assign({}, DEFAULT_SETTINGS, userSettings ? userSettings : NEW_USER_SETTINGS);
                        return [2 /*return*/];
                }
            });
        });
    };
    CycleThroughPanes.prototype.saveSettings = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.saveData(this.settings)];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    return CycleThroughPanes;
}(obsidian.Plugin));

module.exports = CycleThroughPanes;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWFpbi5qcyIsInNvdXJjZXMiOlsibm9kZV9tb2R1bGVzL3RzbGliL3RzbGliLmVzNi5qcyIsInNyYy9tb2RhbC50cyIsInNyYy9zZXR0aW5nc1RhYi50cyIsInNyYy90eXBlcy50cyIsInNyYy9tYWluLnRzIl0sInNvdXJjZXNDb250ZW50IjpbIi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcclxuQ29weXJpZ2h0IChjKSBNaWNyb3NvZnQgQ29ycG9yYXRpb24uXHJcblxyXG5QZXJtaXNzaW9uIHRvIHVzZSwgY29weSwgbW9kaWZ5LCBhbmQvb3IgZGlzdHJpYnV0ZSB0aGlzIHNvZnR3YXJlIGZvciBhbnlcclxucHVycG9zZSB3aXRoIG9yIHdpdGhvdXQgZmVlIGlzIGhlcmVieSBncmFudGVkLlxyXG5cclxuVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEIFwiQVMgSVNcIiBBTkQgVEhFIEFVVEhPUiBESVNDTEFJTVMgQUxMIFdBUlJBTlRJRVMgV0lUSFxyXG5SRUdBUkQgVE8gVEhJUyBTT0ZUV0FSRSBJTkNMVURJTkcgQUxMIElNUExJRUQgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFlcclxuQU5EIEZJVE5FU1MuIElOIE5PIEVWRU5UIFNIQUxMIFRIRSBBVVRIT1IgQkUgTElBQkxFIEZPUiBBTlkgU1BFQ0lBTCwgRElSRUNULFxyXG5JTkRJUkVDVCwgT1IgQ09OU0VRVUVOVElBTCBEQU1BR0VTIE9SIEFOWSBEQU1BR0VTIFdIQVRTT0VWRVIgUkVTVUxUSU5HIEZST01cclxuTE9TUyBPRiBVU0UsIERBVEEgT1IgUFJPRklUUywgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIE5FR0xJR0VOQ0UgT1JcclxuT1RIRVIgVE9SVElPVVMgQUNUSU9OLCBBUklTSU5HIE9VVCBPRiBPUiBJTiBDT05ORUNUSU9OIFdJVEggVEhFIFVTRSBPUlxyXG5QRVJGT1JNQU5DRSBPRiBUSElTIFNPRlRXQVJFLlxyXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiAqL1xyXG4vKiBnbG9iYWwgUmVmbGVjdCwgUHJvbWlzZSwgU3VwcHJlc3NlZEVycm9yLCBTeW1ib2wgKi9cclxuXHJcbnZhciBleHRlbmRTdGF0aWNzID0gZnVuY3Rpb24oZCwgYikge1xyXG4gICAgZXh0ZW5kU3RhdGljcyA9IE9iamVjdC5zZXRQcm90b3R5cGVPZiB8fFxyXG4gICAgICAgICh7IF9fcHJvdG9fXzogW10gfSBpbnN0YW5jZW9mIEFycmF5ICYmIGZ1bmN0aW9uIChkLCBiKSB7IGQuX19wcm90b19fID0gYjsgfSkgfHxcclxuICAgICAgICBmdW5jdGlvbiAoZCwgYikgeyBmb3IgKHZhciBwIGluIGIpIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwoYiwgcCkpIGRbcF0gPSBiW3BdOyB9O1xyXG4gICAgcmV0dXJuIGV4dGVuZFN0YXRpY3MoZCwgYik7XHJcbn07XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19leHRlbmRzKGQsIGIpIHtcclxuICAgIGlmICh0eXBlb2YgYiAhPT0gXCJmdW5jdGlvblwiICYmIGIgIT09IG51bGwpXHJcbiAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcihcIkNsYXNzIGV4dGVuZHMgdmFsdWUgXCIgKyBTdHJpbmcoYikgKyBcIiBpcyBub3QgYSBjb25zdHJ1Y3RvciBvciBudWxsXCIpO1xyXG4gICAgZXh0ZW5kU3RhdGljcyhkLCBiKTtcclxuICAgIGZ1bmN0aW9uIF9fKCkgeyB0aGlzLmNvbnN0cnVjdG9yID0gZDsgfVxyXG4gICAgZC5wcm90b3R5cGUgPSBiID09PSBudWxsID8gT2JqZWN0LmNyZWF0ZShiKSA6IChfXy5wcm90b3R5cGUgPSBiLnByb3RvdHlwZSwgbmV3IF9fKCkpO1xyXG59XHJcblxyXG5leHBvcnQgdmFyIF9fYXNzaWduID0gZnVuY3Rpb24oKSB7XHJcbiAgICBfX2Fzc2lnbiA9IE9iamVjdC5hc3NpZ24gfHwgZnVuY3Rpb24gX19hc3NpZ24odCkge1xyXG4gICAgICAgIGZvciAodmFyIHMsIGkgPSAxLCBuID0gYXJndW1lbnRzLmxlbmd0aDsgaSA8IG47IGkrKykge1xyXG4gICAgICAgICAgICBzID0gYXJndW1lbnRzW2ldO1xyXG4gICAgICAgICAgICBmb3IgKHZhciBwIGluIHMpIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocywgcCkpIHRbcF0gPSBzW3BdO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdDtcclxuICAgIH1cclxuICAgIHJldHVybiBfX2Fzc2lnbi5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19yZXN0KHMsIGUpIHtcclxuICAgIHZhciB0ID0ge307XHJcbiAgICBmb3IgKHZhciBwIGluIHMpIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocywgcCkgJiYgZS5pbmRleE9mKHApIDwgMClcclxuICAgICAgICB0W3BdID0gc1twXTtcclxuICAgIGlmIChzICE9IG51bGwgJiYgdHlwZW9mIE9iamVjdC5nZXRPd25Qcm9wZXJ0eVN5bWJvbHMgPT09IFwiZnVuY3Rpb25cIilcclxuICAgICAgICBmb3IgKHZhciBpID0gMCwgcCA9IE9iamVjdC5nZXRPd25Qcm9wZXJ0eVN5bWJvbHMocyk7IGkgPCBwLmxlbmd0aDsgaSsrKSB7XHJcbiAgICAgICAgICAgIGlmIChlLmluZGV4T2YocFtpXSkgPCAwICYmIE9iamVjdC5wcm90b3R5cGUucHJvcGVydHlJc0VudW1lcmFibGUuY2FsbChzLCBwW2ldKSlcclxuICAgICAgICAgICAgICAgIHRbcFtpXV0gPSBzW3BbaV1dO1xyXG4gICAgICAgIH1cclxuICAgIHJldHVybiB0O1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19kZWNvcmF0ZShkZWNvcmF0b3JzLCB0YXJnZXQsIGtleSwgZGVzYykge1xyXG4gICAgdmFyIGMgPSBhcmd1bWVudHMubGVuZ3RoLCByID0gYyA8IDMgPyB0YXJnZXQgOiBkZXNjID09PSBudWxsID8gZGVzYyA9IE9iamVjdC5nZXRPd25Qcm9wZXJ0eURlc2NyaXB0b3IodGFyZ2V0LCBrZXkpIDogZGVzYywgZDtcclxuICAgIGlmICh0eXBlb2YgUmVmbGVjdCA9PT0gXCJvYmplY3RcIiAmJiB0eXBlb2YgUmVmbGVjdC5kZWNvcmF0ZSA9PT0gXCJmdW5jdGlvblwiKSByID0gUmVmbGVjdC5kZWNvcmF0ZShkZWNvcmF0b3JzLCB0YXJnZXQsIGtleSwgZGVzYyk7XHJcbiAgICBlbHNlIGZvciAodmFyIGkgPSBkZWNvcmF0b3JzLmxlbmd0aCAtIDE7IGkgPj0gMDsgaS0tKSBpZiAoZCA9IGRlY29yYXRvcnNbaV0pIHIgPSAoYyA8IDMgPyBkKHIpIDogYyA+IDMgPyBkKHRhcmdldCwga2V5LCByKSA6IGQodGFyZ2V0LCBrZXkpKSB8fCByO1xyXG4gICAgcmV0dXJuIGMgPiAzICYmIHIgJiYgT2JqZWN0LmRlZmluZVByb3BlcnR5KHRhcmdldCwga2V5LCByKSwgcjtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fcGFyYW0ocGFyYW1JbmRleCwgZGVjb3JhdG9yKSB7XHJcbiAgICByZXR1cm4gZnVuY3Rpb24gKHRhcmdldCwga2V5KSB7IGRlY29yYXRvcih0YXJnZXQsIGtleSwgcGFyYW1JbmRleCk7IH1cclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZXNEZWNvcmF0ZShjdG9yLCBkZXNjcmlwdG9ySW4sIGRlY29yYXRvcnMsIGNvbnRleHRJbiwgaW5pdGlhbGl6ZXJzLCBleHRyYUluaXRpYWxpemVycykge1xyXG4gICAgZnVuY3Rpb24gYWNjZXB0KGYpIHsgaWYgKGYgIT09IHZvaWQgMCAmJiB0eXBlb2YgZiAhPT0gXCJmdW5jdGlvblwiKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiRnVuY3Rpb24gZXhwZWN0ZWRcIik7IHJldHVybiBmOyB9XHJcbiAgICB2YXIga2luZCA9IGNvbnRleHRJbi5raW5kLCBrZXkgPSBraW5kID09PSBcImdldHRlclwiID8gXCJnZXRcIiA6IGtpbmQgPT09IFwic2V0dGVyXCIgPyBcInNldFwiIDogXCJ2YWx1ZVwiO1xyXG4gICAgdmFyIHRhcmdldCA9ICFkZXNjcmlwdG9ySW4gJiYgY3RvciA/IGNvbnRleHRJbltcInN0YXRpY1wiXSA/IGN0b3IgOiBjdG9yLnByb3RvdHlwZSA6IG51bGw7XHJcbiAgICB2YXIgZGVzY3JpcHRvciA9IGRlc2NyaXB0b3JJbiB8fCAodGFyZ2V0ID8gT2JqZWN0LmdldE93blByb3BlcnR5RGVzY3JpcHRvcih0YXJnZXQsIGNvbnRleHRJbi5uYW1lKSA6IHt9KTtcclxuICAgIHZhciBfLCBkb25lID0gZmFsc2U7XHJcbiAgICBmb3IgKHZhciBpID0gZGVjb3JhdG9ycy5sZW5ndGggLSAxOyBpID49IDA7IGktLSkge1xyXG4gICAgICAgIHZhciBjb250ZXh0ID0ge307XHJcbiAgICAgICAgZm9yICh2YXIgcCBpbiBjb250ZXh0SW4pIGNvbnRleHRbcF0gPSBwID09PSBcImFjY2Vzc1wiID8ge30gOiBjb250ZXh0SW5bcF07XHJcbiAgICAgICAgZm9yICh2YXIgcCBpbiBjb250ZXh0SW4uYWNjZXNzKSBjb250ZXh0LmFjY2Vzc1twXSA9IGNvbnRleHRJbi5hY2Nlc3NbcF07XHJcbiAgICAgICAgY29udGV4dC5hZGRJbml0aWFsaXplciA9IGZ1bmN0aW9uIChmKSB7IGlmIChkb25lKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiQ2Fubm90IGFkZCBpbml0aWFsaXplcnMgYWZ0ZXIgZGVjb3JhdGlvbiBoYXMgY29tcGxldGVkXCIpOyBleHRyYUluaXRpYWxpemVycy5wdXNoKGFjY2VwdChmIHx8IG51bGwpKTsgfTtcclxuICAgICAgICB2YXIgcmVzdWx0ID0gKDAsIGRlY29yYXRvcnNbaV0pKGtpbmQgPT09IFwiYWNjZXNzb3JcIiA/IHsgZ2V0OiBkZXNjcmlwdG9yLmdldCwgc2V0OiBkZXNjcmlwdG9yLnNldCB9IDogZGVzY3JpcHRvcltrZXldLCBjb250ZXh0KTtcclxuICAgICAgICBpZiAoa2luZCA9PT0gXCJhY2Nlc3NvclwiKSB7XHJcbiAgICAgICAgICAgIGlmIChyZXN1bHQgPT09IHZvaWQgMCkgY29udGludWU7XHJcbiAgICAgICAgICAgIGlmIChyZXN1bHQgPT09IG51bGwgfHwgdHlwZW9mIHJlc3VsdCAhPT0gXCJvYmplY3RcIikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIk9iamVjdCBleHBlY3RlZFwiKTtcclxuICAgICAgICAgICAgaWYgKF8gPSBhY2NlcHQocmVzdWx0LmdldCkpIGRlc2NyaXB0b3IuZ2V0ID0gXztcclxuICAgICAgICAgICAgaWYgKF8gPSBhY2NlcHQocmVzdWx0LnNldCkpIGRlc2NyaXB0b3Iuc2V0ID0gXztcclxuICAgICAgICAgICAgaWYgKF8gPSBhY2NlcHQocmVzdWx0LmluaXQpKSBpbml0aWFsaXplcnMudW5zaGlmdChfKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgZWxzZSBpZiAoXyA9IGFjY2VwdChyZXN1bHQpKSB7XHJcbiAgICAgICAgICAgIGlmIChraW5kID09PSBcImZpZWxkXCIpIGluaXRpYWxpemVycy51bnNoaWZ0KF8pO1xyXG4gICAgICAgICAgICBlbHNlIGRlc2NyaXB0b3Jba2V5XSA9IF87XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgaWYgKHRhcmdldCkgT2JqZWN0LmRlZmluZVByb3BlcnR5KHRhcmdldCwgY29udGV4dEluLm5hbWUsIGRlc2NyaXB0b3IpO1xyXG4gICAgZG9uZSA9IHRydWU7XHJcbn07XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19ydW5Jbml0aWFsaXplcnModGhpc0FyZywgaW5pdGlhbGl6ZXJzLCB2YWx1ZSkge1xyXG4gICAgdmFyIHVzZVZhbHVlID0gYXJndW1lbnRzLmxlbmd0aCA+IDI7XHJcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IGluaXRpYWxpemVycy5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgIHZhbHVlID0gdXNlVmFsdWUgPyBpbml0aWFsaXplcnNbaV0uY2FsbCh0aGlzQXJnLCB2YWx1ZSkgOiBpbml0aWFsaXplcnNbaV0uY2FsbCh0aGlzQXJnKTtcclxuICAgIH1cclxuICAgIHJldHVybiB1c2VWYWx1ZSA/IHZhbHVlIDogdm9pZCAwO1xyXG59O1xyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fcHJvcEtleSh4KSB7XHJcbiAgICByZXR1cm4gdHlwZW9mIHggPT09IFwic3ltYm9sXCIgPyB4IDogXCJcIi5jb25jYXQoeCk7XHJcbn07XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19zZXRGdW5jdGlvbk5hbWUoZiwgbmFtZSwgcHJlZml4KSB7XHJcbiAgICBpZiAodHlwZW9mIG5hbWUgPT09IFwic3ltYm9sXCIpIG5hbWUgPSBuYW1lLmRlc2NyaXB0aW9uID8gXCJbXCIuY29uY2F0KG5hbWUuZGVzY3JpcHRpb24sIFwiXVwiKSA6IFwiXCI7XHJcbiAgICByZXR1cm4gT2JqZWN0LmRlZmluZVByb3BlcnR5KGYsIFwibmFtZVwiLCB7IGNvbmZpZ3VyYWJsZTogdHJ1ZSwgdmFsdWU6IHByZWZpeCA/IFwiXCIuY29uY2F0KHByZWZpeCwgXCIgXCIsIG5hbWUpIDogbmFtZSB9KTtcclxufTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX21ldGFkYXRhKG1ldGFkYXRhS2V5LCBtZXRhZGF0YVZhbHVlKSB7XHJcbiAgICBpZiAodHlwZW9mIFJlZmxlY3QgPT09IFwib2JqZWN0XCIgJiYgdHlwZW9mIFJlZmxlY3QubWV0YWRhdGEgPT09IFwiZnVuY3Rpb25cIikgcmV0dXJuIFJlZmxlY3QubWV0YWRhdGEobWV0YWRhdGFLZXksIG1ldGFkYXRhVmFsdWUpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hd2FpdGVyKHRoaXNBcmcsIF9hcmd1bWVudHMsIFAsIGdlbmVyYXRvcikge1xyXG4gICAgZnVuY3Rpb24gYWRvcHQodmFsdWUpIHsgcmV0dXJuIHZhbHVlIGluc3RhbmNlb2YgUCA/IHZhbHVlIDogbmV3IFAoZnVuY3Rpb24gKHJlc29sdmUpIHsgcmVzb2x2ZSh2YWx1ZSk7IH0pOyB9XHJcbiAgICByZXR1cm4gbmV3IChQIHx8IChQID0gUHJvbWlzZSkpKGZ1bmN0aW9uIChyZXNvbHZlLCByZWplY3QpIHtcclxuICAgICAgICBmdW5jdGlvbiBmdWxmaWxsZWQodmFsdWUpIHsgdHJ5IHsgc3RlcChnZW5lcmF0b3IubmV4dCh2YWx1ZSkpOyB9IGNhdGNoIChlKSB7IHJlamVjdChlKTsgfSB9XHJcbiAgICAgICAgZnVuY3Rpb24gcmVqZWN0ZWQodmFsdWUpIHsgdHJ5IHsgc3RlcChnZW5lcmF0b3JbXCJ0aHJvd1wiXSh2YWx1ZSkpOyB9IGNhdGNoIChlKSB7IHJlamVjdChlKTsgfSB9XHJcbiAgICAgICAgZnVuY3Rpb24gc3RlcChyZXN1bHQpIHsgcmVzdWx0LmRvbmUgPyByZXNvbHZlKHJlc3VsdC52YWx1ZSkgOiBhZG9wdChyZXN1bHQudmFsdWUpLnRoZW4oZnVsZmlsbGVkLCByZWplY3RlZCk7IH1cclxuICAgICAgICBzdGVwKChnZW5lcmF0b3IgPSBnZW5lcmF0b3IuYXBwbHkodGhpc0FyZywgX2FyZ3VtZW50cyB8fCBbXSkpLm5leHQoKSk7XHJcbiAgICB9KTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZ2VuZXJhdG9yKHRoaXNBcmcsIGJvZHkpIHtcclxuICAgIHZhciBfID0geyBsYWJlbDogMCwgc2VudDogZnVuY3Rpb24oKSB7IGlmICh0WzBdICYgMSkgdGhyb3cgdFsxXTsgcmV0dXJuIHRbMV07IH0sIHRyeXM6IFtdLCBvcHM6IFtdIH0sIGYsIHksIHQsIGc7XHJcbiAgICByZXR1cm4gZyA9IHsgbmV4dDogdmVyYigwKSwgXCJ0aHJvd1wiOiB2ZXJiKDEpLCBcInJldHVyblwiOiB2ZXJiKDIpIH0sIHR5cGVvZiBTeW1ib2wgPT09IFwiZnVuY3Rpb25cIiAmJiAoZ1tTeW1ib2wuaXRlcmF0b3JdID0gZnVuY3Rpb24oKSB7IHJldHVybiB0aGlzOyB9KSwgZztcclxuICAgIGZ1bmN0aW9uIHZlcmIobikgeyByZXR1cm4gZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHN0ZXAoW24sIHZdKTsgfTsgfVxyXG4gICAgZnVuY3Rpb24gc3RlcChvcCkge1xyXG4gICAgICAgIGlmIChmKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiR2VuZXJhdG9yIGlzIGFscmVhZHkgZXhlY3V0aW5nLlwiKTtcclxuICAgICAgICB3aGlsZSAoZyAmJiAoZyA9IDAsIG9wWzBdICYmIChfID0gMCkpLCBfKSB0cnkge1xyXG4gICAgICAgICAgICBpZiAoZiA9IDEsIHkgJiYgKHQgPSBvcFswXSAmIDIgPyB5W1wicmV0dXJuXCJdIDogb3BbMF0gPyB5W1widGhyb3dcIl0gfHwgKCh0ID0geVtcInJldHVyblwiXSkgJiYgdC5jYWxsKHkpLCAwKSA6IHkubmV4dCkgJiYgISh0ID0gdC5jYWxsKHksIG9wWzFdKSkuZG9uZSkgcmV0dXJuIHQ7XHJcbiAgICAgICAgICAgIGlmICh5ID0gMCwgdCkgb3AgPSBbb3BbMF0gJiAyLCB0LnZhbHVlXTtcclxuICAgICAgICAgICAgc3dpdGNoIChvcFswXSkge1xyXG4gICAgICAgICAgICAgICAgY2FzZSAwOiBjYXNlIDE6IHQgPSBvcDsgYnJlYWs7XHJcbiAgICAgICAgICAgICAgICBjYXNlIDQ6IF8ubGFiZWwrKzsgcmV0dXJuIHsgdmFsdWU6IG9wWzFdLCBkb25lOiBmYWxzZSB9O1xyXG4gICAgICAgICAgICAgICAgY2FzZSA1OiBfLmxhYmVsKys7IHkgPSBvcFsxXTsgb3AgPSBbMF07IGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgY2FzZSA3OiBvcCA9IF8ub3BzLnBvcCgpOyBfLnRyeXMucG9wKCk7IGNvbnRpbnVlO1xyXG4gICAgICAgICAgICAgICAgZGVmYXVsdDpcclxuICAgICAgICAgICAgICAgICAgICBpZiAoISh0ID0gXy50cnlzLCB0ID0gdC5sZW5ndGggPiAwICYmIHRbdC5sZW5ndGggLSAxXSkgJiYgKG9wWzBdID09PSA2IHx8IG9wWzBdID09PSAyKSkgeyBfID0gMDsgY29udGludWU7IH1cclxuICAgICAgICAgICAgICAgICAgICBpZiAob3BbMF0gPT09IDMgJiYgKCF0IHx8IChvcFsxXSA+IHRbMF0gJiYgb3BbMV0gPCB0WzNdKSkpIHsgXy5sYWJlbCA9IG9wWzFdOyBicmVhazsgfVxyXG4gICAgICAgICAgICAgICAgICAgIGlmIChvcFswXSA9PT0gNiAmJiBfLmxhYmVsIDwgdFsxXSkgeyBfLmxhYmVsID0gdFsxXTsgdCA9IG9wOyBicmVhazsgfVxyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0ICYmIF8ubGFiZWwgPCB0WzJdKSB7IF8ubGFiZWwgPSB0WzJdOyBfLm9wcy5wdXNoKG9wKTsgYnJlYWs7IH1cclxuICAgICAgICAgICAgICAgICAgICBpZiAodFsyXSkgXy5vcHMucG9wKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgXy50cnlzLnBvcCgpOyBjb250aW51ZTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBvcCA9IGJvZHkuY2FsbCh0aGlzQXJnLCBfKTtcclxuICAgICAgICB9IGNhdGNoIChlKSB7IG9wID0gWzYsIGVdOyB5ID0gMDsgfSBmaW5hbGx5IHsgZiA9IHQgPSAwOyB9XHJcbiAgICAgICAgaWYgKG9wWzBdICYgNSkgdGhyb3cgb3BbMV07IHJldHVybiB7IHZhbHVlOiBvcFswXSA/IG9wWzFdIDogdm9pZCAwLCBkb25lOiB0cnVlIH07XHJcbiAgICB9XHJcbn1cclxuXHJcbmV4cG9ydCB2YXIgX19jcmVhdGVCaW5kaW5nID0gT2JqZWN0LmNyZWF0ZSA/IChmdW5jdGlvbihvLCBtLCBrLCBrMikge1xyXG4gICAgaWYgKGsyID09PSB1bmRlZmluZWQpIGsyID0gaztcclxuICAgIHZhciBkZXNjID0gT2JqZWN0LmdldE93blByb3BlcnR5RGVzY3JpcHRvcihtLCBrKTtcclxuICAgIGlmICghZGVzYyB8fCAoXCJnZXRcIiBpbiBkZXNjID8gIW0uX19lc01vZHVsZSA6IGRlc2Mud3JpdGFibGUgfHwgZGVzYy5jb25maWd1cmFibGUpKSB7XHJcbiAgICAgICAgZGVzYyA9IHsgZW51bWVyYWJsZTogdHJ1ZSwgZ2V0OiBmdW5jdGlvbigpIHsgcmV0dXJuIG1ba107IH0gfTtcclxuICAgIH1cclxuICAgIE9iamVjdC5kZWZpbmVQcm9wZXJ0eShvLCBrMiwgZGVzYyk7XHJcbn0pIDogKGZ1bmN0aW9uKG8sIG0sIGssIGsyKSB7XHJcbiAgICBpZiAoazIgPT09IHVuZGVmaW5lZCkgazIgPSBrO1xyXG4gICAgb1trMl0gPSBtW2tdO1xyXG59KTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2V4cG9ydFN0YXIobSwgbykge1xyXG4gICAgZm9yICh2YXIgcCBpbiBtKSBpZiAocCAhPT0gXCJkZWZhdWx0XCIgJiYgIU9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvLCBwKSkgX19jcmVhdGVCaW5kaW5nKG8sIG0sIHApO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX192YWx1ZXMobykge1xyXG4gICAgdmFyIHMgPSB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgU3ltYm9sLml0ZXJhdG9yLCBtID0gcyAmJiBvW3NdLCBpID0gMDtcclxuICAgIGlmIChtKSByZXR1cm4gbS5jYWxsKG8pO1xyXG4gICAgaWYgKG8gJiYgdHlwZW9mIG8ubGVuZ3RoID09PSBcIm51bWJlclwiKSByZXR1cm4ge1xyXG4gICAgICAgIG5leHQ6IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgaWYgKG8gJiYgaSA+PSBvLmxlbmd0aCkgbyA9IHZvaWQgMDtcclxuICAgICAgICAgICAgcmV0dXJuIHsgdmFsdWU6IG8gJiYgb1tpKytdLCBkb25lOiAhbyB9O1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB0aHJvdyBuZXcgVHlwZUVycm9yKHMgPyBcIk9iamVjdCBpcyBub3QgaXRlcmFibGUuXCIgOiBcIlN5bWJvbC5pdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3JlYWQobywgbikge1xyXG4gICAgdmFyIG0gPSB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgb1tTeW1ib2wuaXRlcmF0b3JdO1xyXG4gICAgaWYgKCFtKSByZXR1cm4gbztcclxuICAgIHZhciBpID0gbS5jYWxsKG8pLCByLCBhciA9IFtdLCBlO1xyXG4gICAgdHJ5IHtcclxuICAgICAgICB3aGlsZSAoKG4gPT09IHZvaWQgMCB8fCBuLS0gPiAwKSAmJiAhKHIgPSBpLm5leHQoKSkuZG9uZSkgYXIucHVzaChyLnZhbHVlKTtcclxuICAgIH1cclxuICAgIGNhdGNoIChlcnJvcikgeyBlID0geyBlcnJvcjogZXJyb3IgfTsgfVxyXG4gICAgZmluYWxseSB7XHJcbiAgICAgICAgdHJ5IHtcclxuICAgICAgICAgICAgaWYgKHIgJiYgIXIuZG9uZSAmJiAobSA9IGlbXCJyZXR1cm5cIl0pKSBtLmNhbGwoaSk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGZpbmFsbHkgeyBpZiAoZSkgdGhyb3cgZS5lcnJvcjsgfVxyXG4gICAgfVxyXG4gICAgcmV0dXJuIGFyO1xyXG59XHJcblxyXG4vKiogQGRlcHJlY2F0ZWQgKi9cclxuZXhwb3J0IGZ1bmN0aW9uIF9fc3ByZWFkKCkge1xyXG4gICAgZm9yICh2YXIgYXIgPSBbXSwgaSA9IDA7IGkgPCBhcmd1bWVudHMubGVuZ3RoOyBpKyspXHJcbiAgICAgICAgYXIgPSBhci5jb25jYXQoX19yZWFkKGFyZ3VtZW50c1tpXSkpO1xyXG4gICAgcmV0dXJuIGFyO1xyXG59XHJcblxyXG4vKiogQGRlcHJlY2F0ZWQgKi9cclxuZXhwb3J0IGZ1bmN0aW9uIF9fc3ByZWFkQXJyYXlzKCkge1xyXG4gICAgZm9yICh2YXIgcyA9IDAsIGkgPSAwLCBpbCA9IGFyZ3VtZW50cy5sZW5ndGg7IGkgPCBpbDsgaSsrKSBzICs9IGFyZ3VtZW50c1tpXS5sZW5ndGg7XHJcbiAgICBmb3IgKHZhciByID0gQXJyYXkocyksIGsgPSAwLCBpID0gMDsgaSA8IGlsOyBpKyspXHJcbiAgICAgICAgZm9yICh2YXIgYSA9IGFyZ3VtZW50c1tpXSwgaiA9IDAsIGpsID0gYS5sZW5ndGg7IGogPCBqbDsgaisrLCBrKyspXHJcbiAgICAgICAgICAgIHJba10gPSBhW2pdO1xyXG4gICAgcmV0dXJuIHI7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3NwcmVhZEFycmF5KHRvLCBmcm9tLCBwYWNrKSB7XHJcbiAgICBpZiAocGFjayB8fCBhcmd1bWVudHMubGVuZ3RoID09PSAyKSBmb3IgKHZhciBpID0gMCwgbCA9IGZyb20ubGVuZ3RoLCBhcjsgaSA8IGw7IGkrKykge1xyXG4gICAgICAgIGlmIChhciB8fCAhKGkgaW4gZnJvbSkpIHtcclxuICAgICAgICAgICAgaWYgKCFhcikgYXIgPSBBcnJheS5wcm90b3R5cGUuc2xpY2UuY2FsbChmcm9tLCAwLCBpKTtcclxuICAgICAgICAgICAgYXJbaV0gPSBmcm9tW2ldO1xyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIHJldHVybiB0by5jb25jYXQoYXIgfHwgQXJyYXkucHJvdG90eXBlLnNsaWNlLmNhbGwoZnJvbSkpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hd2FpdCh2KSB7XHJcbiAgICByZXR1cm4gdGhpcyBpbnN0YW5jZW9mIF9fYXdhaXQgPyAodGhpcy52ID0gdiwgdGhpcykgOiBuZXcgX19hd2FpdCh2KTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fYXN5bmNHZW5lcmF0b3IodGhpc0FyZywgX2FyZ3VtZW50cywgZ2VuZXJhdG9yKSB7XHJcbiAgICBpZiAoIVN5bWJvbC5hc3luY0l0ZXJhdG9yKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiU3ltYm9sLmFzeW5jSXRlcmF0b3IgaXMgbm90IGRlZmluZWQuXCIpO1xyXG4gICAgdmFyIGcgPSBnZW5lcmF0b3IuYXBwbHkodGhpc0FyZywgX2FyZ3VtZW50cyB8fCBbXSksIGksIHEgPSBbXTtcclxuICAgIHJldHVybiBpID0ge30sIHZlcmIoXCJuZXh0XCIpLCB2ZXJiKFwidGhyb3dcIiksIHZlcmIoXCJyZXR1cm5cIiksIGlbU3ltYm9sLmFzeW5jSXRlcmF0b3JdID0gZnVuY3Rpb24gKCkgeyByZXR1cm4gdGhpczsgfSwgaTtcclxuICAgIGZ1bmN0aW9uIHZlcmIobikgeyBpZiAoZ1tuXSkgaVtuXSA9IGZ1bmN0aW9uICh2KSB7IHJldHVybiBuZXcgUHJvbWlzZShmdW5jdGlvbiAoYSwgYikgeyBxLnB1c2goW24sIHYsIGEsIGJdKSA+IDEgfHwgcmVzdW1lKG4sIHYpOyB9KTsgfTsgfVxyXG4gICAgZnVuY3Rpb24gcmVzdW1lKG4sIHYpIHsgdHJ5IHsgc3RlcChnW25dKHYpKTsgfSBjYXRjaCAoZSkgeyBzZXR0bGUocVswXVszXSwgZSk7IH0gfVxyXG4gICAgZnVuY3Rpb24gc3RlcChyKSB7IHIudmFsdWUgaW5zdGFuY2VvZiBfX2F3YWl0ID8gUHJvbWlzZS5yZXNvbHZlKHIudmFsdWUudikudGhlbihmdWxmaWxsLCByZWplY3QpIDogc2V0dGxlKHFbMF1bMl0sIHIpOyB9XHJcbiAgICBmdW5jdGlvbiBmdWxmaWxsKHZhbHVlKSB7IHJlc3VtZShcIm5leHRcIiwgdmFsdWUpOyB9XHJcbiAgICBmdW5jdGlvbiByZWplY3QodmFsdWUpIHsgcmVzdW1lKFwidGhyb3dcIiwgdmFsdWUpOyB9XHJcbiAgICBmdW5jdGlvbiBzZXR0bGUoZiwgdikgeyBpZiAoZih2KSwgcS5zaGlmdCgpLCBxLmxlbmd0aCkgcmVzdW1lKHFbMF1bMF0sIHFbMF1bMV0pOyB9XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2FzeW5jRGVsZWdhdG9yKG8pIHtcclxuICAgIHZhciBpLCBwO1xyXG4gICAgcmV0dXJuIGkgPSB7fSwgdmVyYihcIm5leHRcIiksIHZlcmIoXCJ0aHJvd1wiLCBmdW5jdGlvbiAoZSkgeyB0aHJvdyBlOyB9KSwgdmVyYihcInJldHVyblwiKSwgaVtTeW1ib2wuaXRlcmF0b3JdID0gZnVuY3Rpb24gKCkgeyByZXR1cm4gdGhpczsgfSwgaTtcclxuICAgIGZ1bmN0aW9uIHZlcmIobiwgZikgeyBpW25dID0gb1tuXSA/IGZ1bmN0aW9uICh2KSB7IHJldHVybiAocCA9ICFwKSA/IHsgdmFsdWU6IF9fYXdhaXQob1tuXSh2KSksIGRvbmU6IGZhbHNlIH0gOiBmID8gZih2KSA6IHY7IH0gOiBmOyB9XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2FzeW5jVmFsdWVzKG8pIHtcclxuICAgIGlmICghU3ltYm9sLmFzeW5jSXRlcmF0b3IpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJTeW1ib2wuYXN5bmNJdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbiAgICB2YXIgbSA9IG9bU3ltYm9sLmFzeW5jSXRlcmF0b3JdLCBpO1xyXG4gICAgcmV0dXJuIG0gPyBtLmNhbGwobykgOiAobyA9IHR5cGVvZiBfX3ZhbHVlcyA9PT0gXCJmdW5jdGlvblwiID8gX192YWx1ZXMobykgOiBvW1N5bWJvbC5pdGVyYXRvcl0oKSwgaSA9IHt9LCB2ZXJiKFwibmV4dFwiKSwgdmVyYihcInRocm93XCIpLCB2ZXJiKFwicmV0dXJuXCIpLCBpW1N5bWJvbC5hc3luY0l0ZXJhdG9yXSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHRoaXM7IH0sIGkpO1xyXG4gICAgZnVuY3Rpb24gdmVyYihuKSB7IGlbbl0gPSBvW25dICYmIGZ1bmN0aW9uICh2KSB7IHJldHVybiBuZXcgUHJvbWlzZShmdW5jdGlvbiAocmVzb2x2ZSwgcmVqZWN0KSB7IHYgPSBvW25dKHYpLCBzZXR0bGUocmVzb2x2ZSwgcmVqZWN0LCB2LmRvbmUsIHYudmFsdWUpOyB9KTsgfTsgfVxyXG4gICAgZnVuY3Rpb24gc2V0dGxlKHJlc29sdmUsIHJlamVjdCwgZCwgdikgeyBQcm9taXNlLnJlc29sdmUodikudGhlbihmdW5jdGlvbih2KSB7IHJlc29sdmUoeyB2YWx1ZTogdiwgZG9uZTogZCB9KTsgfSwgcmVqZWN0KTsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19tYWtlVGVtcGxhdGVPYmplY3QoY29va2VkLCByYXcpIHtcclxuICAgIGlmIChPYmplY3QuZGVmaW5lUHJvcGVydHkpIHsgT2JqZWN0LmRlZmluZVByb3BlcnR5KGNvb2tlZCwgXCJyYXdcIiwgeyB2YWx1ZTogcmF3IH0pOyB9IGVsc2UgeyBjb29rZWQucmF3ID0gcmF3OyB9XHJcbiAgICByZXR1cm4gY29va2VkO1xyXG59O1xyXG5cclxudmFyIF9fc2V0TW9kdWxlRGVmYXVsdCA9IE9iamVjdC5jcmVhdGUgPyAoZnVuY3Rpb24obywgdikge1xyXG4gICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KG8sIFwiZGVmYXVsdFwiLCB7IGVudW1lcmFibGU6IHRydWUsIHZhbHVlOiB2IH0pO1xyXG59KSA6IGZ1bmN0aW9uKG8sIHYpIHtcclxuICAgIG9bXCJkZWZhdWx0XCJdID0gdjtcclxufTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2ltcG9ydFN0YXIobW9kKSB7XHJcbiAgICBpZiAobW9kICYmIG1vZC5fX2VzTW9kdWxlKSByZXR1cm4gbW9kO1xyXG4gICAgdmFyIHJlc3VsdCA9IHt9O1xyXG4gICAgaWYgKG1vZCAhPSBudWxsKSBmb3IgKHZhciBrIGluIG1vZCkgaWYgKGsgIT09IFwiZGVmYXVsdFwiICYmIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChtb2QsIGspKSBfX2NyZWF0ZUJpbmRpbmcocmVzdWx0LCBtb2QsIGspO1xyXG4gICAgX19zZXRNb2R1bGVEZWZhdWx0KHJlc3VsdCwgbW9kKTtcclxuICAgIHJldHVybiByZXN1bHQ7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2ltcG9ydERlZmF1bHQobW9kKSB7XHJcbiAgICByZXR1cm4gKG1vZCAmJiBtb2QuX19lc01vZHVsZSkgPyBtb2QgOiB7IGRlZmF1bHQ6IG1vZCB9O1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19jbGFzc1ByaXZhdGVGaWVsZEdldChyZWNlaXZlciwgc3RhdGUsIGtpbmQsIGYpIHtcclxuICAgIGlmIChraW5kID09PSBcImFcIiAmJiAhZikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlByaXZhdGUgYWNjZXNzb3Igd2FzIGRlZmluZWQgd2l0aG91dCBhIGdldHRlclwiKTtcclxuICAgIGlmICh0eXBlb2Ygc3RhdGUgPT09IFwiZnVuY3Rpb25cIiA/IHJlY2VpdmVyICE9PSBzdGF0ZSB8fCAhZiA6ICFzdGF0ZS5oYXMocmVjZWl2ZXIpKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiQ2Fubm90IHJlYWQgcHJpdmF0ZSBtZW1iZXIgZnJvbSBhbiBvYmplY3Qgd2hvc2UgY2xhc3MgZGlkIG5vdCBkZWNsYXJlIGl0XCIpO1xyXG4gICAgcmV0dXJuIGtpbmQgPT09IFwibVwiID8gZiA6IGtpbmQgPT09IFwiYVwiID8gZi5jYWxsKHJlY2VpdmVyKSA6IGYgPyBmLnZhbHVlIDogc3RhdGUuZ2V0KHJlY2VpdmVyKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fY2xhc3NQcml2YXRlRmllbGRTZXQocmVjZWl2ZXIsIHN0YXRlLCB2YWx1ZSwga2luZCwgZikge1xyXG4gICAgaWYgKGtpbmQgPT09IFwibVwiKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiUHJpdmF0ZSBtZXRob2QgaXMgbm90IHdyaXRhYmxlXCIpO1xyXG4gICAgaWYgKGtpbmQgPT09IFwiYVwiICYmICFmKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiUHJpdmF0ZSBhY2Nlc3NvciB3YXMgZGVmaW5lZCB3aXRob3V0IGEgc2V0dGVyXCIpO1xyXG4gICAgaWYgKHR5cGVvZiBzdGF0ZSA9PT0gXCJmdW5jdGlvblwiID8gcmVjZWl2ZXIgIT09IHN0YXRlIHx8ICFmIDogIXN0YXRlLmhhcyhyZWNlaXZlcikpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJDYW5ub3Qgd3JpdGUgcHJpdmF0ZSBtZW1iZXIgdG8gYW4gb2JqZWN0IHdob3NlIGNsYXNzIGRpZCBub3QgZGVjbGFyZSBpdFwiKTtcclxuICAgIHJldHVybiAoa2luZCA9PT0gXCJhXCIgPyBmLmNhbGwocmVjZWl2ZXIsIHZhbHVlKSA6IGYgPyBmLnZhbHVlID0gdmFsdWUgOiBzdGF0ZS5zZXQocmVjZWl2ZXIsIHZhbHVlKSksIHZhbHVlO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19jbGFzc1ByaXZhdGVGaWVsZEluKHN0YXRlLCByZWNlaXZlcikge1xyXG4gICAgaWYgKHJlY2VpdmVyID09PSBudWxsIHx8ICh0eXBlb2YgcmVjZWl2ZXIgIT09IFwib2JqZWN0XCIgJiYgdHlwZW9mIHJlY2VpdmVyICE9PSBcImZ1bmN0aW9uXCIpKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiQ2Fubm90IHVzZSAnaW4nIG9wZXJhdG9yIG9uIG5vbi1vYmplY3RcIik7XHJcbiAgICByZXR1cm4gdHlwZW9mIHN0YXRlID09PSBcImZ1bmN0aW9uXCIgPyByZWNlaXZlciA9PT0gc3RhdGUgOiBzdGF0ZS5oYXMocmVjZWl2ZXIpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hZGREaXNwb3NhYmxlUmVzb3VyY2UoZW52LCB2YWx1ZSwgYXN5bmMpIHtcclxuICAgIGlmICh2YWx1ZSAhPT0gbnVsbCAmJiB2YWx1ZSAhPT0gdm9pZCAwKSB7XHJcbiAgICAgICAgaWYgKHR5cGVvZiB2YWx1ZSAhPT0gXCJvYmplY3RcIiAmJiB0eXBlb2YgdmFsdWUgIT09IFwiZnVuY3Rpb25cIikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIk9iamVjdCBleHBlY3RlZC5cIik7XHJcbiAgICAgICAgdmFyIGRpc3Bvc2U7XHJcbiAgICAgICAgaWYgKGFzeW5jKSB7XHJcbiAgICAgICAgICAgIGlmICghU3ltYm9sLmFzeW5jRGlzcG9zZSkgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlN5bWJvbC5hc3luY0Rpc3Bvc2UgaXMgbm90IGRlZmluZWQuXCIpO1xyXG4gICAgICAgICAgICBkaXNwb3NlID0gdmFsdWVbU3ltYm9sLmFzeW5jRGlzcG9zZV07XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGlmIChkaXNwb3NlID09PSB2b2lkIDApIHtcclxuICAgICAgICAgICAgaWYgKCFTeW1ib2wuZGlzcG9zZSkgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlN5bWJvbC5kaXNwb3NlIGlzIG5vdCBkZWZpbmVkLlwiKTtcclxuICAgICAgICAgICAgZGlzcG9zZSA9IHZhbHVlW1N5bWJvbC5kaXNwb3NlXTtcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKHR5cGVvZiBkaXNwb3NlICE9PSBcImZ1bmN0aW9uXCIpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJPYmplY3Qgbm90IGRpc3Bvc2FibGUuXCIpO1xyXG4gICAgICAgIGVudi5zdGFjay5wdXNoKHsgdmFsdWU6IHZhbHVlLCBkaXNwb3NlOiBkaXNwb3NlLCBhc3luYzogYXN5bmMgfSk7XHJcbiAgICB9XHJcbiAgICBlbHNlIGlmIChhc3luYykge1xyXG4gICAgICAgIGVudi5zdGFjay5wdXNoKHsgYXN5bmM6IHRydWUgfSk7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gdmFsdWU7XHJcbn1cclxuXHJcbnZhciBfU3VwcHJlc3NlZEVycm9yID0gdHlwZW9mIFN1cHByZXNzZWRFcnJvciA9PT0gXCJmdW5jdGlvblwiID8gU3VwcHJlc3NlZEVycm9yIDogZnVuY3Rpb24gKGVycm9yLCBzdXBwcmVzc2VkLCBtZXNzYWdlKSB7XHJcbiAgICB2YXIgZSA9IG5ldyBFcnJvcihtZXNzYWdlKTtcclxuICAgIHJldHVybiBlLm5hbWUgPSBcIlN1cHByZXNzZWRFcnJvclwiLCBlLmVycm9yID0gZXJyb3IsIGUuc3VwcHJlc3NlZCA9IHN1cHByZXNzZWQsIGU7XHJcbn07XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19kaXNwb3NlUmVzb3VyY2VzKGVudikge1xyXG4gICAgZnVuY3Rpb24gZmFpbChlKSB7XHJcbiAgICAgICAgZW52LmVycm9yID0gZW52Lmhhc0Vycm9yID8gbmV3IF9TdXBwcmVzc2VkRXJyb3IoZSwgZW52LmVycm9yLCBcIkFuIGVycm9yIHdhcyBzdXBwcmVzc2VkIGR1cmluZyBkaXNwb3NhbC5cIikgOiBlO1xyXG4gICAgICAgIGVudi5oYXNFcnJvciA9IHRydWU7XHJcbiAgICB9XHJcbiAgICBmdW5jdGlvbiBuZXh0KCkge1xyXG4gICAgICAgIHdoaWxlIChlbnYuc3RhY2subGVuZ3RoKSB7XHJcbiAgICAgICAgICAgIHZhciByZWMgPSBlbnYuc3RhY2sucG9wKCk7XHJcbiAgICAgICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgICAgICB2YXIgcmVzdWx0ID0gcmVjLmRpc3Bvc2UgJiYgcmVjLmRpc3Bvc2UuY2FsbChyZWMudmFsdWUpO1xyXG4gICAgICAgICAgICAgICAgaWYgKHJlYy5hc3luYykgcmV0dXJuIFByb21pc2UucmVzb2x2ZShyZXN1bHQpLnRoZW4obmV4dCwgZnVuY3Rpb24oZSkgeyBmYWlsKGUpOyByZXR1cm4gbmV4dCgpOyB9KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBjYXRjaCAoZSkge1xyXG4gICAgICAgICAgICAgICAgZmFpbChlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBpZiAoZW52Lmhhc0Vycm9yKSB0aHJvdyBlbnYuZXJyb3I7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gbmV4dCgpO1xyXG59XHJcblxyXG5leHBvcnQgZGVmYXVsdCB7XHJcbiAgICBfX2V4dGVuZHM6IF9fZXh0ZW5kcyxcclxuICAgIF9fYXNzaWduOiBfX2Fzc2lnbixcclxuICAgIF9fcmVzdDogX19yZXN0LFxyXG4gICAgX19kZWNvcmF0ZTogX19kZWNvcmF0ZSxcclxuICAgIF9fcGFyYW06IF9fcGFyYW0sXHJcbiAgICBfX21ldGFkYXRhOiBfX21ldGFkYXRhLFxyXG4gICAgX19hd2FpdGVyOiBfX2F3YWl0ZXIsXHJcbiAgICBfX2dlbmVyYXRvcjogX19nZW5lcmF0b3IsXHJcbiAgICBfX2NyZWF0ZUJpbmRpbmc6IF9fY3JlYXRlQmluZGluZyxcclxuICAgIF9fZXhwb3J0U3RhcjogX19leHBvcnRTdGFyLFxyXG4gICAgX192YWx1ZXM6IF9fdmFsdWVzLFxyXG4gICAgX19yZWFkOiBfX3JlYWQsXHJcbiAgICBfX3NwcmVhZDogX19zcHJlYWQsXHJcbiAgICBfX3NwcmVhZEFycmF5czogX19zcHJlYWRBcnJheXMsXHJcbiAgICBfX3NwcmVhZEFycmF5OiBfX3NwcmVhZEFycmF5LFxyXG4gICAgX19hd2FpdDogX19hd2FpdCxcclxuICAgIF9fYXN5bmNHZW5lcmF0b3I6IF9fYXN5bmNHZW5lcmF0b3IsXHJcbiAgICBfX2FzeW5jRGVsZWdhdG9yOiBfX2FzeW5jRGVsZWdhdG9yLFxyXG4gICAgX19hc3luY1ZhbHVlczogX19hc3luY1ZhbHVlcyxcclxuICAgIF9fbWFrZVRlbXBsYXRlT2JqZWN0OiBfX21ha2VUZW1wbGF0ZU9iamVjdCxcclxuICAgIF9faW1wb3J0U3RhcjogX19pbXBvcnRTdGFyLFxyXG4gICAgX19pbXBvcnREZWZhdWx0OiBfX2ltcG9ydERlZmF1bHQsXHJcbiAgICBfX2NsYXNzUHJpdmF0ZUZpZWxkR2V0OiBfX2NsYXNzUHJpdmF0ZUZpZWxkR2V0LFxyXG4gICAgX19jbGFzc1ByaXZhdGVGaWVsZFNldDogX19jbGFzc1ByaXZhdGVGaWVsZFNldCxcclxuICAgIF9fY2xhc3NQcml2YXRlRmllbGRJbjogX19jbGFzc1ByaXZhdGVGaWVsZEluLFxyXG4gICAgX19hZGREaXNwb3NhYmxlUmVzb3VyY2U6IF9fYWRkRGlzcG9zYWJsZVJlc291cmNlLFxyXG4gICAgX19kaXNwb3NlUmVzb3VyY2VzOiBfX2Rpc3Bvc2VSZXNvdXJjZXMsXHJcbn07XHJcbiIsImltcG9ydCB7IFN1Z2dlc3RNb2RhbCwgV29ya3NwYWNlTGVhZiB9IGZyb20gXCJvYnNpZGlhblwiO1xuaW1wb3J0IEN5Y2xlVGhyb3VnaFBhbmVzIGZyb20gXCIuL21haW5cIjtcblxuZXhwb3J0IGNsYXNzIEdlbmVyYWxNb2RhbCBleHRlbmRzIFN1Z2dlc3RNb2RhbDxzdHJpbmc+IHtcbiAgICByZXNvbHZlOiAodmFsdWU6IG51bWJlcikgPT4gdm9pZDtcblxuICAgIGNvbnN0cnVjdG9yKFxuICAgICAgICBwcml2YXRlIGxlYXZlczogV29ya3NwYWNlTGVhZltdLFxuICAgICAgICBwcml2YXRlIHJlYWRvbmx5IHBsdWdpbjogQ3ljbGVUaHJvdWdoUGFuZXNcbiAgICApIHtcbiAgICAgICAgc3VwZXIoYXBwKTtcbiAgICB9XG5cbiAgICBvcGVuKCk6IFByb21pc2U8bnVtYmVyPiB7XG4gICAgICAgIHRoaXMuZGltQmFja2dyb3VuZCA9IGZhbHNlO1xuICAgICAgICBzdXBlci5vcGVuKCk7XG5cbiAgICAgICAgdGhpcy5jaG9vc2VyLnNldFNlbGVjdGVkSXRlbSgxKTtcbiAgICAgICAgdGhpcy5mb2N1c1RhYigpO1xuXG4gICAgICAgIHRoaXMuY29udGFpbmVyRWxcbiAgICAgICAgICAgIC5nZXRFbGVtZW50c0J5Q2xhc3NOYW1lKFwicHJvbXB0LWlucHV0LWNvbnRhaW5lclwiKVxuICAgICAgICAgICAgLml0ZW0oMClcbiAgICAgICAgICAgIC5kZXRhY2goKTtcblxuICAgICAgICAvLyBob3RrZXkgPSB0aGlzLmFwcC5ob3RrZXlNYW5hZ2VyLmJha2VkSWRzLmZpbmQoKGUpPT5lID09IFwiXCIpXG5cbiAgICAgICAgdGhpcy5zY29wZS5yZWdpc3RlcihbXCJDdHJsXCJdLCBcIlRhYlwiLCAoZSkgPT4ge1xuICAgICAgICAgICAgdGhpcy5jaG9vc2VyLnNldFNlbGVjdGVkSXRlbSh0aGlzLmNob29zZXIuc2VsZWN0ZWRJdGVtICsgMSk7XG4gICAgICAgICAgICB0aGlzLmZvY3VzVGFiKCk7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIHRoaXMuc2NvcGUucmVnaXN0ZXIoW1wiQ3RybFwiLCBcIlNoaWZ0XCJdLCBcIlRhYlwiLCAoZSkgPT4ge1xuICAgICAgICAgICAgdGhpcy5jaG9vc2VyLnNldFNlbGVjdGVkSXRlbSh0aGlzLmNob29zZXIuc2VsZWN0ZWRJdGVtIC0gMSk7XG4gICAgICAgICAgICB0aGlzLmZvY3VzVGFiKCk7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIHJldHVybiBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4ge1xuICAgICAgICAgICAgdGhpcy5yZXNvbHZlID0gcmVzb2x2ZTtcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgb25DbG9zZSgpIHtcbiAgICAgICAgaWYgKHRoaXMucmVzb2x2ZSkgdGhpcy5yZXNvbHZlKHRoaXMuY2hvb3Nlci5zZWxlY3RlZEl0ZW0pO1xuICAgIH1cblxuICAgIGdldFN1Z2dlc3Rpb25zKHF1ZXJ5OiBzdHJpbmcpOiBzdHJpbmdbXSB7XG4gICAgICAgIHJldHVybiB0aGlzLmxlYXZlcy5tYXAoKGxlYWYpID0+IGxlYWYudmlldy5nZXREaXNwbGF5VGV4dCgpKTtcbiAgICB9XG5cbiAgICByZW5kZXJTdWdnZXN0aW9uKHZhbHVlOiBzdHJpbmcsIGVsOiBIVE1MRWxlbWVudCk6IHZvaWQge1xuICAgICAgICBlbC5zZXRUZXh0KHZhbHVlKTtcbiAgICB9XG5cbiAgICBvbkNob29zZVN1Z2dlc3Rpb24oaXRlbTogc3RyaW5nLCBldnQ6IE1vdXNlRXZlbnQgfCBLZXlib2FyZEV2ZW50KSB7fVxuXG4gICAgZm9jdXNUYWIoKTogdm9pZCB7XG4gICAgICAgIHRoaXMucGx1Z2luLnF1ZXVlRm9jdXNMZWFmKHRoaXMubGVhdmVzW3RoaXMuY2hvb3Nlci5zZWxlY3RlZEl0ZW1dKTtcbiAgICB9XG59XG4iLCJpbXBvcnQgeyBQbHVnaW5TZXR0aW5nVGFiLCBTZXR0aW5nIH0gZnJvbSBcIm9ic2lkaWFuXCI7XG5pbXBvcnQgQ3ljbGVUaHJvdWdoUGFuZXMgZnJvbSBcIi4vbWFpblwiO1xuaW1wb3J0IHsgU2V0dGluZ3MgfSBmcm9tIFwiLi90eXBlc1wiO1xuXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBDVFBTZXR0aW5nVGFiIGV4dGVuZHMgUGx1Z2luU2V0dGluZ1RhYiB7XG4gICAgc2V0dGluZ3M6IFNldHRpbmdzO1xuICAgIHBsdWdpbjogQ3ljbGVUaHJvdWdoUGFuZXM7XG5cbiAgICBjb25zdHJ1Y3RvcihwbHVnaW46IEN5Y2xlVGhyb3VnaFBhbmVzLCBzZXR0aW5nczogU2V0dGluZ3MpIHtcbiAgICAgICAgc3VwZXIocGx1Z2luLmFwcCwgcGx1Z2luKTtcbiAgICAgICAgdGhpcy5zZXR0aW5ncyA9IHNldHRpbmdzO1xuICAgICAgICB0aGlzLnBsdWdpbiA9IHBsdWdpbjtcbiAgICB9XG5cbiAgICBkaXNwbGF5KCkge1xuICAgICAgICBjb25zdCB7IGNvbnRhaW5lckVsIH0gPSB0aGlzO1xuXG4gICAgICAgIGNvbnRhaW5lckVsLmVtcHR5KCk7XG4gICAgICAgIGNvbnRhaW5lckVsLmNyZWF0ZUVsKFwiaDJcIiwge1xuICAgICAgICAgICAgdGV4dDogXCJDeWNsZSB0aHJvdWdoIFBhbmVzIENvbmZpZ3VyYXRpb25cIixcbiAgICAgICAgfSk7XG5cbiAgICAgICAgbmV3IFNldHRpbmcoY29udGFpbmVyRWwpXG4gICAgICAgICAgICAuc2V0TmFtZShcIk9ubHkgY3ljbGUgdGhyb3VnaCB0YWJzIHdpdGggc3BlY2lmaWMgVmlldyBUeXBlc1wiKVxuICAgICAgICAgICAgLmFkZFRvZ2dsZSgoY2IpID0+IHtcbiAgICAgICAgICAgICAgICBjYi5zZXRWYWx1ZSh0aGlzLnNldHRpbmdzLnVzZVZpZXdUeXBlcyk7XG4gICAgICAgICAgICAgICAgY2Iub25DaGFuZ2UoYXN5bmMgKHZhbHVlKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMuc2V0dGluZ3MudXNlVmlld1R5cGVzID0gdmFsdWU7XG4gICAgICAgICAgICAgICAgICAgIGF3YWl0IHRoaXMucGx1Z2luLnNhdmVTZXR0aW5ncygpO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfSk7XG5cbiAgICAgICAgY29uc3QgZGVzY0VsID0gY3JlYXRlRnJhZ21lbnQoKTtcbiAgICAgICAgZGVzY0VsLmFwcGVuZChcbiAgICAgICAgICAgIGNyZWF0ZUVsKFwicFwiLCB7XG4gICAgICAgICAgICAgICAgdGV4dDogXCJJZiB0aGUgb3B0aW9uIGFib3ZlIGlzIGVuYWJsZWQ6IFRoZXNlIGFyZSB0aGUgVmlldyBUeXBlcyB0aGlzIFBsdWdpbiB3aWxsIGN5Y2xlIHRocm91Z2ggdXNpbmcgYW55IG9mIHRoZSBhdmFpbGFibGUgY29tbWFuZHMuXCIsXG4gICAgICAgICAgICB9KSxcbiAgICAgICAgICAgIGNyZWF0ZUVsKFwicFwiLCB7XG4gICAgICAgICAgICAgICAgdGV4dDogJ1RvIGFkZCBhIG5ldyBWaWV3IFR5cGUgdG8gdGhpcyBMaXN0LCBzaW1wbHkgcnVuIHRoZSBDb21tYW5kOiBcIkN5Y2xlIHRocm91Z2ggUGFuZXM6IEVuYWJsZSB0aGlzIFZpZXcgVHlwZVwiLiBNb3JlIGFkdmFuY2VkIFVzZXJzIGNhbiBlZGl0IGFuZCBkZWxldGUgdGhlIFR5cGVzIGluIHRoZSB0ZXh0IGZpZWxkIChvbmUgcGVyIGxpbmUpLicsXG4gICAgICAgICAgICB9KVxuICAgICAgICApO1xuXG4gICAgICAgIG5ldyBTZXR0aW5nKGNvbnRhaW5lckVsKVxuICAgICAgICAgICAgLnNldE5hbWUoXCJFbmFibGVkIFZpZXcgVHlwZXNcIilcbiAgICAgICAgICAgIC5zZXREZXNjKGRlc2NFbClcbiAgICAgICAgICAgIC5hZGRUZXh0QXJlYSgoY2IpID0+IHtcbiAgICAgICAgICAgICAgICBsZXQgdmFsdWUgPSBcIlwiO1xuICAgICAgICAgICAgICAgIHRoaXMuc2V0dGluZ3Mudmlld1R5cGVzLmZvckVhY2goXG4gICAgICAgICAgICAgICAgICAgICh0eXBlKSA9PiAodmFsdWUgKz0gdHlwZSArIFwiXFxuXCIpXG4gICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICBjYi5zZXRWYWx1ZSh2YWx1ZSk7XG4gICAgICAgICAgICAgICAgY2Iuc2V0UGxhY2Vob2xkZXIoXCJtYXJrZG93blwiKTtcbiAgICAgICAgICAgICAgICBjYi5vbkNoYW5nZShhc3luYyAobmV3VmFsdWUpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgLy8gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTm8gZW1wdHkgbGluZXNcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5zZXR0aW5ncy52aWV3VHlwZXMgPSBuZXdWYWx1ZVxuICAgICAgICAgICAgICAgICAgICAgICAgLnNwbGl0KFwiXFxuXCIpXG4gICAgICAgICAgICAgICAgICAgICAgICAuZmlsdGVyKChwcmUpID0+ICEhcHJlKTtcbiAgICAgICAgICAgICAgICAgICAgYXdhaXQgdGhpcy5wbHVnaW4uc2F2ZVNldHRpbmdzKCk7XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9KTtcblxuICAgICAgICBuZXcgU2V0dGluZyhjb250YWluZXJFbClcbiAgICAgICAgICAgIC5zZXROYW1lKFwiU2hvdyBtb2RhbCB3aGVuIHN3aXRjaGluZyB0YWJzXCIpXG4gICAgICAgICAgICAuYWRkVG9nZ2xlKChjYikgPT4ge1xuICAgICAgICAgICAgICAgIGNiLnNldFZhbHVlKHRoaXMuc2V0dGluZ3Muc2hvd01vZGFsKTtcbiAgICAgICAgICAgICAgICBjYi5vbkNoYW5nZShhc3luYyAodmFsdWUpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5zZXR0aW5ncy5zaG93TW9kYWwgPSB2YWx1ZTtcbiAgICAgICAgICAgICAgICAgICAgYXdhaXQgdGhpcy5wbHVnaW4uc2F2ZVNldHRpbmdzKCk7XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9KTtcblxuICAgICAgICBuZXcgU2V0dGluZyhjb250YWluZXJFbClcbiAgICAgICAgICAgIC5zZXROYW1lKFwiRm9jdXMgdGFiIG9uIHJlbGVhc2VcIilcbiAgICAgICAgICAgIC5zZXREZXNjKFxuICAgICAgICAgICAgICAgIFwiSWYgZW5hYmxlZCwgZGVmZXIgc3dpdGNoaW5nIHRhYnMgdW50aWwgdGhlIGN0cmwga2V5IGlzIHJlbGVhc2VkLCBzaW1pbGFyIHRvIFZTIENvZGUgYW5kIEZpcmVmb3hcIlxuICAgICAgICAgICAgKVxuICAgICAgICAgICAgLmFkZFRvZ2dsZSgoY2IpID0+IHtcbiAgICAgICAgICAgICAgICBjYi5zZXRWYWx1ZSh0aGlzLnNldHRpbmdzLmZvY3VzTGVhZk9uS2V5VXApO1xuICAgICAgICAgICAgICAgIGNiLm9uQ2hhbmdlKGFzeW5jICh2YWx1ZSkgPT4ge1xuICAgICAgICAgICAgICAgICAgICB0aGlzLnNldHRpbmdzLmZvY3VzTGVhZk9uS2V5VXAgPSB2YWx1ZTtcbiAgICAgICAgICAgICAgICAgICAgYXdhaXQgdGhpcy5wbHVnaW4uc2F2ZVNldHRpbmdzKCk7XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICB9KTtcblxuICAgICAgICBuZXcgU2V0dGluZyhjb250YWluZXJFbCkuc2V0TmFtZShcIlNraXAgcGlubmVkIHRhYnNcIikuYWRkVG9nZ2xlKChjYikgPT4ge1xuICAgICAgICAgICAgY2Iuc2V0VmFsdWUodGhpcy5zZXR0aW5ncy5za2lwUGlubmVkKTtcbiAgICAgICAgICAgIGNiLm9uQ2hhbmdlKGFzeW5jICh2YWx1ZSkgPT4ge1xuICAgICAgICAgICAgICAgIHRoaXMuc2V0dGluZ3Muc2tpcFBpbm5lZCA9IHZhbHVlO1xuICAgICAgICAgICAgICAgIGF3YWl0IHRoaXMucGx1Z2luLnNhdmVTZXR0aW5ncygpO1xuICAgICAgICAgICAgfSk7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIG5ldyBTZXR0aW5nKGNvbnRhaW5lckVsKVxuICAgICAgICAgICAgLnNldE5hbWUoXCJTdGF5IGluIGN1cnJlbnQgc3BsaXRcIilcbiAgICAgICAgICAgIC5zZXREZXNjKFxuICAgICAgICAgICAgICAgIFwiSWYgZW5hYmxlZCBhbmQgdGhlIGN1cnJlbnRseSBhY3RpdmUgZmlsZSBpcyBpbiB0aGUgc2lkZWJhciwgeW91IGN5Y2xlIHdpdGhpbiB0aGF0IHNpZGViYXIgYW5kIGNhbid0IHN3aXRjaCB0byB0aGUgbWFpbiB0YWJzLiBVc2UgdGhlIFwiXG4gICAgICAgICAgICApXG4gICAgICAgICAgICAuYWRkVG9nZ2xlKChjYikgPT4ge1xuICAgICAgICAgICAgICAgIGNiLnNldFZhbHVlKHRoaXMuc2V0dGluZ3Muc3RheUluU3BsaXQpO1xuICAgICAgICAgICAgICAgIGNiLm9uQ2hhbmdlKGFzeW5jICh2YWx1ZSkgPT4ge1xuICAgICAgICAgICAgICAgICAgICB0aGlzLnNldHRpbmdzLnN0YXlJblNwbGl0ID0gdmFsdWU7XG4gICAgICAgICAgICAgICAgICAgIGF3YWl0IHRoaXMucGx1Z2luLnNhdmVTZXR0aW5ncygpO1xuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgfSk7XG4gICAgfVxufVxuIiwiZXhwb3J0IGludGVyZmFjZSBTZXR0aW5ncyB7XG4gICAgdmlld1R5cGVzOiBzdHJpbmdbXTtcbiAgICBzaG93TW9kYWw6IGJvb2xlYW47XG4gICAgc2tpcFBpbm5lZDogYm9vbGVhbjtcbiAgICBzdGF5SW5TcGxpdDogYm9vbGVhbjtcbiAgICBmb2N1c0xlYWZPbktleVVwOiBib29sZWFuO1xuICAgIHVzZVZpZXdUeXBlczogYm9vbGVhbjtcbn1cblxuZXhwb3J0IGNvbnN0IERFRkFVTFRfU0VUVElOR1M6IFNldHRpbmdzID0ge1xuICAgIHZpZXdUeXBlczogW1wibWFya2Rvd25cIiwgXCJjYW52YXNcIiwgXCJwZGZcIl0sXG4gICAgc2hvd01vZGFsOiB0cnVlLFxuICAgIHNraXBQaW5uZWQ6IGZhbHNlLFxuICAgIHN0YXlJblNwbGl0OiB0cnVlLFxuICAgIGZvY3VzTGVhZk9uS2V5VXA6IGZhbHNlLCAvLyBvcHQtaW4gZm9yIGV4aXN0aW5nIHVzZXJzXG4gICAgdXNlVmlld1R5cGVzOiB0cnVlLFxufTtcblxuZXhwb3J0IGNvbnN0IE5FV19VU0VSX1NFVFRJTkdTOiBQYXJ0aWFsPFNldHRpbmdzPiA9IHtcbiAgICBmb2N1c0xlYWZPbktleVVwOiB0cnVlLCAvLyBkZWZhdWx0IGZvciBuZXcgdXNlcnNcbiAgICB1c2VWaWV3VHlwZXM6IGZhbHNlLFxufTtcblxuZGVjbGFyZSBtb2R1bGUgXCJvYnNpZGlhblwiIHtcbiAgICBpbnRlcmZhY2UgQXBwIHtcbiAgICAgICAgaG90a2V5TWFuYWdlcjoge1xuICAgICAgICAgICAgYmFrZWRJZHM6IHN0cmluZ1tdO1xuICAgICAgICAgICAgYmFrZWRIb3RrZXlzOiB7IG1vZGlmaWVyczogc3RyaW5nOyBrZXk6IHN0cmluZyB9W107XG4gICAgICAgIH07XG4gICAgfVxuXG4gICAgaW50ZXJmYWNlIFdvcmtzcGFjZUxlYWYge1xuICAgICAgICBhY3RpdmVUaW1lOiBudW1iZXI7XG4gICAgfVxuXG4gICAgaW50ZXJmYWNlIFdvcmtzcGFjZUl0ZW0ge1xuICAgICAgICBvcGVuTGVhZihsZWFmOiBXb3Jrc3BhY2VMZWFmKTogdm9pZDtcbiAgICB9XG5cbiAgICBpbnRlcmZhY2UgTW9kYWwge1xuICAgICAgICBjaG9vc2VyOiB7XG4gICAgICAgICAgICBtb3ZlRG93bjogYW55O1xuICAgICAgICAgICAgbW92ZVVwOiBhbnk7XG4gICAgICAgICAgICBzZWxlY3RlZEl0ZW06IG51bWJlcjtcbiAgICAgICAgICAgIHNldFNlbGVjdGVkSXRlbTogKGluZGV4OiBudW1iZXIpID0+IHZvaWQ7XG4gICAgICAgIH07XG4gICAgICAgIGRpbUJhY2tncm91bmQ6IGJvb2xlYW47XG4gICAgfVxufVxuIiwiaW1wb3J0IHsgUGxhdGZvcm0sIFBsdWdpbiwgV29ya3NwYWNlTGVhZiB9IGZyb20gXCJvYnNpZGlhblwiO1xuaW1wb3J0IHsgR2VuZXJhbE1vZGFsIH0gZnJvbSBcIi4vbW9kYWxcIjtcbmltcG9ydCBDVFBTZXR0aW5nVGFiIGZyb20gXCIuL3NldHRpbmdzVGFiXCI7XG5pbXBvcnQgeyBERUZBVUxUX1NFVFRJTkdTLCBORVdfVVNFUl9TRVRUSU5HUywgU2V0dGluZ3MgfSBmcm9tIFwiLi90eXBlc1wiO1xuXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBDeWNsZVRocm91Z2hQYW5lcyBleHRlbmRzIFBsdWdpbiB7XG4gICAgc2V0dGluZ3M6IFNldHRpbmdzO1xuICAgIGN0cmxQcmVzc2VkVGltZXN0YW1wID0gMDtcbiAgICBjdHJsS2V5Q29kZTogc3RyaW5nIHwgdW5kZWZpbmVkO1xuICAgIHF1ZXVlZEZvY3VzTGVhZjogV29ya3NwYWNlTGVhZjtcbiAgICBsZWFmSW5kZXggPSAwO1xuICAgIG1vZGFsOiBHZW5lcmFsTW9kYWwgfCB1bmRlZmluZWQ7XG4gICAgbGVhdmVzOiBXb3Jrc3BhY2VMZWFmW107XG5cbiAgICBrZXlEb3duRnVuYyA9IHRoaXMub25LZXlEb3duLmJpbmQodGhpcyk7XG4gICAga2V5VXBGdW5jID0gdGhpcy5vbktleVVwLmJpbmQodGhpcyk7XG5cbiAgICBnZXRMZWF2ZXNPZlR5cGVzKHR5cGVzOiBzdHJpbmdbXSk6IFdvcmtzcGFjZUxlYWZbXSB7XG4gICAgICAgIGNvbnN0IGxlYXZlczogV29ya3NwYWNlTGVhZltdID0gW107XG4gICAgICAgIGNvbnN0IGFjdGl2ZUxlYWYgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICAgICAgdGhpcy5hcHAud29ya3NwYWNlLml0ZXJhdGVBbGxMZWF2ZXMoKGxlYWYpID0+IHtcbiAgICAgICAgICAgIGlmICh0aGlzLnNldHRpbmdzLnNraXBQaW5uZWQgJiYgbGVhZi5nZXRWaWV3U3RhdGUoKS5waW5uZWQpIHJldHVybjtcblxuICAgICAgICAgICAgY29uc3QgY29ycmVjdFZpZXdUeXBlID1cbiAgICAgICAgICAgICAgICAhdGhpcy5zZXR0aW5ncy51c2VWaWV3VHlwZXMgfHxcbiAgICAgICAgICAgICAgICB0eXBlcy5jb250YWlucyhsZWFmLnZpZXcuZ2V0Vmlld1R5cGUoKSk7XG5cbiAgICAgICAgICAgIGlmICghY29ycmVjdFZpZXdUeXBlKSByZXR1cm47XG5cbiAgICAgICAgICAgIGNvbnN0IGlzTWFpbldpbmRvdyA9IGxlYWYudmlldy5jb250YWluZXJFbC53aW4gPT0gd2luZG93O1xuICAgICAgICAgICAgY29uc3Qgc2FtZVdpbmRvdyA9IGxlYWYudmlldy5jb250YWluZXJFbC53aW4gPT0gYWN0aXZlV2luZG93O1xuXG4gICAgICAgICAgICBsZXQgY29ycmVjdFBhbmUgPSBmYWxzZTtcbiAgICAgICAgICAgIGlmIChpc01haW5XaW5kb3cpIHtcbiAgICAgICAgICAgICAgICBpZiAodGhpcy5zZXR0aW5ncy5zdGF5SW5TcGxpdCkge1xuICAgICAgICAgICAgICAgICAgICBjb3JyZWN0UGFuZSA9XG4gICAgICAgICAgICAgICAgICAgICAgICBzYW1lV2luZG93ICYmIGxlYWYuZ2V0Um9vdCgpID09IGFjdGl2ZUxlYWYuZ2V0Um9vdCgpO1xuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIGNvcnJlY3RQYW5lID1cbiAgICAgICAgICAgICAgICAgICAgICAgIHNhbWVXaW5kb3cgJiZcbiAgICAgICAgICAgICAgICAgICAgICAgIGxlYWYuZ2V0Um9vdCgpID09IHRoaXMuYXBwLndvcmtzcGFjZS5yb290U3BsaXQ7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBjb3JyZWN0UGFuZSA9IHNhbWVXaW5kb3c7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBpZiAoY29ycmVjdFBhbmUpIHtcbiAgICAgICAgICAgICAgICBsZWF2ZXMucHVzaChsZWFmKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSk7XG5cbiAgICAgICAgcmV0dXJuIGxlYXZlcztcbiAgICB9XG5cbiAgICBhc3luYyBvbmxvYWQoKSB7XG4gICAgICAgIGNvbnNvbGUubG9nKFwibG9hZGluZyBwbHVnaW46IEN5Y2xlIHRocm91Z2ggcGFuZXNcIik7XG5cbiAgICAgICAgYXdhaXQgdGhpcy5sb2FkU2V0dGluZ3MoKTtcblxuICAgICAgICB0aGlzLmFkZFNldHRpbmdUYWIobmV3IENUUFNldHRpbmdUYWIodGhpcywgdGhpcy5zZXR0aW5ncykpO1xuXG4gICAgICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICAgICAgICBpZDogXCJjeWNsZS10aHJvdWdoLXBhbmVzXCIsXG4gICAgICAgICAgICBuYW1lOiBcIkdvIHRvIHJpZ2h0IHRhYlwiLFxuICAgICAgICAgICAgY2hlY2tDYWxsYmFjazogKGNoZWNraW5nOiBib29sZWFuKSA9PiB7XG4gICAgICAgICAgICAgICAgY29uc3QgYWN0aXZlID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG5cbiAgICAgICAgICAgICAgICBpZiAoYWN0aXZlKSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICghY2hlY2tpbmcpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IGxlYXZlczogV29ya3NwYWNlTGVhZltdID0gdGhpcy5nZXRMZWF2ZXNPZlR5cGVzKFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuc2V0dGluZ3Mudmlld1R5cGVzXG4gICAgICAgICAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgaW5kZXggPSBsZWF2ZXMuaW5kZXhPZihhY3RpdmUpO1xuXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoaW5kZXggPT09IGxlYXZlcy5sZW5ndGggLSAxKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5xdWV1ZUZvY3VzTGVhZihsZWF2ZXNbMF0pO1xuICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLnF1ZXVlRm9jdXNMZWFmKGxlYXZlc1tpbmRleCArIDFdKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgICAgfSxcbiAgICAgICAgfSk7XG5cbiAgICAgICAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgICAgICAgIGlkOiBcImN5Y2xlLXRocm91Z2gtcGFuZXMtcmV2ZXJzZVwiLFxuICAgICAgICAgICAgbmFtZTogXCJHbyB0byBsZWZ0IHRhYlwiLFxuICAgICAgICAgICAgY2hlY2tDYWxsYmFjazogKGNoZWNraW5nOiBib29sZWFuKSA9PiB7XG4gICAgICAgICAgICAgICAgY29uc3QgYWN0aXZlID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgICAgICAgICAgICAgaWYgKGFjdGl2ZSkge1xuICAgICAgICAgICAgICAgICAgICBpZiAoIWNoZWNraW5nKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBjb25zdCBsZWF2ZXM6IFdvcmtzcGFjZUxlYWZbXSA9IHRoaXMuZ2V0TGVhdmVzT2ZUeXBlcyhcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLnNldHRpbmdzLnZpZXdUeXBlc1xuICAgICAgICAgICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IGluZGV4ID0gbGVhdmVzLmluZGV4T2YoYWN0aXZlKTtcblxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGluZGV4ICE9PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoaW5kZXggPT09IDApIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5xdWV1ZUZvY3VzTGVhZihsZWF2ZXNbbGVhdmVzLmxlbmd0aCAtIDFdKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLnF1ZXVlRm9jdXNMZWFmKGxlYXZlc1tpbmRleCAtIDFdKTtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH0sXG4gICAgICAgIH0pO1xuXG4gICAgICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICAgICAgICBpZDogXCJjeWNsZS10aHJvdWdoLXBhbmVzLWFkZC12aWV3XCIsXG4gICAgICAgICAgICBuYW1lOiBcIkVuYWJsZSB0aGlzIFZpZXcgVHlwZVwiLFxuICAgICAgICAgICAgY2hlY2tDYWxsYmFjazogKGNoZWNraW5nOiBib29sZWFuKSA9PiB7XG4gICAgICAgICAgICAgICAgY29uc3QgYWN0aXZlID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgICAgICAgICAgICAgaWYgKFxuICAgICAgICAgICAgICAgICAgICBhY3RpdmUgJiZcbiAgICAgICAgICAgICAgICAgICAgIXRoaXMuc2V0dGluZ3Mudmlld1R5cGVzLmNvbnRhaW5zKGFjdGl2ZS52aWV3LmdldFZpZXdUeXBlKCkpXG4gICAgICAgICAgICAgICAgKSB7XG4gICAgICAgICAgICAgICAgICAgIGlmICghY2hlY2tpbmcpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuc2V0dGluZ3Mudmlld1R5cGVzLnB1c2goYWN0aXZlLnZpZXcuZ2V0Vmlld1R5cGUoKSk7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLnNhdmVTZXR0aW5ncygpO1xuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgICAgICB9LFxuICAgICAgICB9KTtcblxuICAgICAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgICAgICAgaWQ6IFwiY3ljbGUtdGhyb3VnaC1wYW5lcy1yZW1vdmUtdmlld1wiLFxuICAgICAgICAgICAgbmFtZTogXCJEaXNhYmxlIHRoaXMgVmlldyBUeXBlXCIsXG4gICAgICAgICAgICBjaGVja0NhbGxiYWNrOiAoY2hlY2tpbmc6IGJvb2xlYW4pID0+IHtcbiAgICAgICAgICAgICAgICBjb25zdCBhY3RpdmUgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICAgICAgICAgICAgICBpZiAoXG4gICAgICAgICAgICAgICAgICAgIGFjdGl2ZSAmJlxuICAgICAgICAgICAgICAgICAgICB0aGlzLnNldHRpbmdzLnZpZXdUeXBlcy5jb250YWlucyhhY3RpdmUudmlldy5nZXRWaWV3VHlwZSgpKVxuICAgICAgICAgICAgICAgICkge1xuICAgICAgICAgICAgICAgICAgICBpZiAoIWNoZWNraW5nKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aGlzLnNldHRpbmdzLnZpZXdUeXBlcy5yZW1vdmUoXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgYWN0aXZlLnZpZXcuZ2V0Vmlld1R5cGUoKVxuICAgICAgICAgICAgICAgICAgICAgICAgKTtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRoaXMuc2F2ZVNldHRpbmdzKCk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgIH0sXG4gICAgICAgIH0pO1xuXG4gICAgICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICAgICAgICBpZDogXCJmb2N1cy1sZWZ0LXNpZGViYXJcIixcbiAgICAgICAgICAgIG5hbWU6IFwiRm9jdXMgb24gbGVmdCBzaWRlYmFyXCIsXG4gICAgICAgICAgICBjYWxsYmFjazogKCkgPT4ge1xuICAgICAgICAgICAgICAgIGFwcC53b3Jrc3BhY2UubGVmdFNwbGl0LmV4cGFuZCgpO1xuICAgICAgICAgICAgICAgIGxldCBsZWFmOiBXb3Jrc3BhY2VMZWFmO1xuICAgICAgICAgICAgICAgIGFwcC53b3Jrc3BhY2UuaXRlcmF0ZUFsbExlYXZlcygoZSkgPT4ge1xuICAgICAgICAgICAgICAgICAgICBpZiAoZS5nZXRSb290KCkgPT0gYXBwLndvcmtzcGFjZS5sZWZ0U3BsaXQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChlLmFjdGl2ZVRpbWUgPiAobGVhZj8uYWN0aXZlVGltZSB8fCAwKSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxlYWYgPSBlO1xuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICAgICAgdGhpcy5xdWV1ZUZvY3VzTGVhZihsZWFmKTtcbiAgICAgICAgICAgIH0sXG4gICAgICAgIH0pO1xuXG4gICAgICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICAgICAgICBpZDogXCJmb2N1cy1yaWdodC1zaWRlYmFyXCIsXG4gICAgICAgICAgICBuYW1lOiBcIkZvY3VzIG9uIHJpZ2h0IHNpZGViYXJcIixcbiAgICAgICAgICAgIGNhbGxiYWNrOiAoKSA9PiB7XG4gICAgICAgICAgICAgICAgYXBwLndvcmtzcGFjZS5yaWdodFNwbGl0LmV4cGFuZCgpO1xuICAgICAgICAgICAgICAgIGxldCBsZWFmOiBXb3Jrc3BhY2VMZWFmO1xuICAgICAgICAgICAgICAgIGFwcC53b3Jrc3BhY2UuaXRlcmF0ZUFsbExlYXZlcygoZSkgPT4ge1xuICAgICAgICAgICAgICAgICAgICBpZiAoZS5nZXRSb290KCkgPT0gYXBwLndvcmtzcGFjZS5yaWdodFNwbGl0KSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAoZS5hY3RpdmVUaW1lID4gKGxlYWY/LmFjdGl2ZVRpbWUgfHwgMCkpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBsZWFmID0gZTtcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgICAgIHRoaXMucXVldWVGb2N1c0xlYWYobGVhZik7XG4gICAgICAgICAgICB9LFxuICAgICAgICB9KTtcblxuICAgICAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgICAgICAgaWQ6IFwiZm9jdXMtb24tbGFzdC1hY3RpdmUtcGFuZVwiLFxuICAgICAgICAgICAgbmFtZTogXCJHbyB0byBwcmV2aW91cyB0YWJcIixcbiAgICAgICAgICAgIGNhbGxiYWNrOiBhc3luYyAoKSA9PiB7XG4gICAgICAgICAgICAgICAgdGhpcy5zZXRMZWF2ZXMoKTtcblxuICAgICAgICAgICAgICAgIHRoaXMubGVhZkluZGV4ID0gKHRoaXMubGVhZkluZGV4ICsgMSkgJSB0aGlzLmxlYXZlcy5sZW5ndGg7XG4gICAgICAgICAgICAgICAgY29uc3QgbGVhZiA9IHRoaXMubGVhdmVzW3RoaXMubGVhZkluZGV4XTtcblxuICAgICAgICAgICAgICAgIGlmIChsZWFmKSB7XG4gICAgICAgICAgICAgICAgICAgIHRoaXMucXVldWVGb2N1c0xlYWYobGVhZik7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSxcbiAgICAgICAgfSk7XG4gICAgICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICAgICAgICBpZDogXCJmb2N1cy1vbi1sYXN0LWFjdGl2ZS1wYW5lLXJldmVyc2VcIixcbiAgICAgICAgICAgIG5hbWU6IFwiR28gdG8gbmV4dCB0YWJcIixcbiAgICAgICAgICAgIGNhbGxiYWNrOiBhc3luYyAoKSA9PiB7XG4gICAgICAgICAgICAgICAgdGhpcy5zZXRMZWF2ZXMoKTtcbiAgICAgICAgICAgICAgICB0aGlzLmxlYWZJbmRleCA9XG4gICAgICAgICAgICAgICAgICAgICh0aGlzLmxlYWZJbmRleCAtIDEgKyB0aGlzLmxlYXZlcy5sZW5ndGgpICVcbiAgICAgICAgICAgICAgICAgICAgdGhpcy5sZWF2ZXMubGVuZ3RoO1xuICAgICAgICAgICAgICAgIGNvbnN0IGxlYWYgPSB0aGlzLmxlYXZlc1t0aGlzLmxlYWZJbmRleF07XG5cbiAgICAgICAgICAgICAgICBpZiAobGVhZikge1xuICAgICAgICAgICAgICAgICAgICB0aGlzLnF1ZXVlRm9jdXNMZWFmKGxlYWYpO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0sXG4gICAgICAgIH0pO1xuXG4gICAgICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKFwia2V5ZG93blwiLCB0aGlzLmtleURvd25GdW5jKTtcbiAgICAgICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoXCJrZXl1cFwiLCB0aGlzLmtleVVwRnVuYyk7XG4gICAgfVxuXG4gICAgcXVldWVGb2N1c0xlYWYobGVhZjogV29ya3NwYWNlTGVhZikge1xuICAgICAgICBpZiAodGhpcy5zZXR0aW5ncy5mb2N1c0xlYWZPbktleVVwKSB7XG4gICAgICAgICAgICB0aGlzLnF1ZXVlZEZvY3VzTGVhZiA9IGxlYWY7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICB0aGlzLmZvY3VzTGVhZihsZWFmKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIGZvY3VzTGVhZihsZWFmOiBXb3Jrc3BhY2VMZWFmKSB7XG4gICAgICAgIGlmIChsZWFmKSB7XG4gICAgICAgICAgICBjb25zdCByb290ID0gbGVhZi5nZXRSb290KCk7XG4gICAgICAgICAgICBpZiAocm9vdCAhPSB0aGlzLmFwcC53b3Jrc3BhY2Uucm9vdFNwbGl0ICYmIFBsYXRmb3JtLmlzTW9iaWxlKSB7XG4gICAgICAgICAgICAgICAgcm9vdC5vcGVuTGVhZihsZWFmKTtcbiAgICAgICAgICAgICAgICBsZWFmLmFjdGl2ZVRpbWUgPSBEYXRlLm5vdygpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICB0aGlzLmFwcC53b3Jrc3BhY2Uuc2V0QWN0aXZlTGVhZihsZWFmLCB7IGZvY3VzOiB0cnVlIH0pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKGxlYWYuZ2V0Vmlld1N0YXRlKCkudHlwZSA9PSBcInNlYXJjaFwiKSB7XG4gICAgICAgICAgICAgICAgY29uc3Qgc2VhcmNoID0gbGVhZi52aWV3LmNvbnRhaW5lckVsLmZpbmQoXG4gICAgICAgICAgICAgICAgICAgIFwiLnNlYXJjaC1pbnB1dC1jb250YWluZXIgaW5wdXRcIlxuICAgICAgICAgICAgICAgICk7XG5cbiAgICAgICAgICAgICAgICBzZWFyY2guZm9jdXMoKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cblxuICAgIHNldExlYXZlcygpIHtcbiAgICAgICAgaWYgKCF0aGlzLmxlYXZlcykge1xuICAgICAgICAgICAgY29uc3QgbGVhdmVzID0gdGhpcy5nZXRMZWF2ZXNPZlR5cGVzKHRoaXMuc2V0dGluZ3Mudmlld1R5cGVzKTtcbiAgICAgICAgICAgIGxlYXZlcy5zb3J0KChhLCBiKSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGIuYWN0aXZlVGltZSAtIGEuYWN0aXZlVGltZTtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgdGhpcy5sZWF2ZXMgPSBsZWF2ZXM7XG4gICAgICAgICAgICB0aGlzLmxlYWZJbmRleCA9IGxlYXZlcy5pbmRleE9mKHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG9uS2V5RG93bihlOiBLZXlib2FyZEV2ZW50KSB7XG4gICAgICAgIGlmIChlLmtleSA9PSBcIkNvbnRyb2xcIikge1xuICAgICAgICAgICAgdGhpcy5jdHJsUHJlc3NlZFRpbWVzdGFtcCA9IGUudGltZVN0YW1wO1xuICAgICAgICAgICAgdGhpcy5jdHJsS2V5Q29kZSA9IGUuY29kZTtcblxuICAgICAgICAgICAgLy8gY2xlYW4gc2xhdGUgLS0gcHJldmVudCBjdHJsIGtleXN0cm9rZSBmcm9tIGFjY2lkZW50YWxseSBzd2l0Y2hpbmcgdG8gYW5vdGhlciB0YWJcbiAgICAgICAgICAgIHRoaXMucXVldWVkRm9jdXNMZWFmID0gdW5kZWZpbmVkO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgb25LZXlVcChlOiBLZXlib2FyZEV2ZW50KSB7XG4gICAgICAgIGlmIChlLmNvZGUgPT0gdGhpcy5jdHJsS2V5Q29kZSAmJiB0aGlzLmN0cmxQcmVzc2VkVGltZXN0YW1wKSB7XG4gICAgICAgICAgICB0aGlzLmN0cmxQcmVzc2VkVGltZXN0YW1wID0gMDtcbiAgICAgICAgICAgIHRoaXMubGVhdmVzID0gbnVsbDtcblxuICAgICAgICAgICAgdGhpcy5tb2RhbD8uY2xvc2UoKTtcblxuICAgICAgICAgICAgaWYgKHRoaXMucXVldWVkRm9jdXNMZWFmKSB7XG4gICAgICAgICAgICAgICAgdGhpcy5mb2N1c0xlYWYodGhpcy5xdWV1ZWRGb2N1c0xlYWYpO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICB0aGlzLm1vZGFsID0gdW5kZWZpbmVkO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYgKFxuICAgICAgICAgICAgZS5jb2RlID09IFwiVGFiXCIgJiZcbiAgICAgICAgICAgIHRoaXMuY3RybFByZXNzZWRUaW1lc3RhbXAgJiZcbiAgICAgICAgICAgIHRoaXMuc2V0dGluZ3Muc2hvd01vZGFsICYmXG4gICAgICAgICAgICAhdGhpcy5tb2RhbCAmJlxuICAgICAgICAgICAgdGhpcy5sZWF2ZXNcbiAgICAgICAgKSB7XG4gICAgICAgICAgICB0aGlzLm1vZGFsID0gbmV3IEdlbmVyYWxNb2RhbCh0aGlzLmxlYXZlcywgdGhpcyk7XG4gICAgICAgICAgICB0aGlzLm1vZGFsLm9wZW4oKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIG9udW5sb2FkKCkge1xuICAgICAgICBjb25zb2xlLmxvZyhcInVubG9hZGluZyBwbHVnaW46IEN5Y2xlIHRocm91Z2ggcGFuZXNcIik7XG4gICAgICAgIHdpbmRvdy5yZW1vdmVFdmVudExpc3RlbmVyKFwia2V5ZG93blwiLCB0aGlzLmtleURvd25GdW5jKTtcbiAgICAgICAgd2luZG93LnJlbW92ZUV2ZW50TGlzdGVuZXIoXCJrZXl1cFwiLCB0aGlzLmtleVVwRnVuYyk7XG4gICAgfVxuXG4gICAgYXN5bmMgbG9hZFNldHRpbmdzKCkge1xuICAgICAgICAvLyByZXR1cm5zIG51bGwgaWYgLm9ic2lkaWFuL3BsdWdpbnMvY3ljbGUtdGhyb3VnaC1wYW5lcy9kYXRhLmpzb24gZG9lcyBub3QgZXhpc3RcbiAgICAgICAgY29uc3QgdXNlclNldHRpbmdzID0gYXdhaXQgdGhpcy5sb2FkRGF0YSgpO1xuXG4gICAgICAgIHRoaXMuc2V0dGluZ3MgPSBPYmplY3QuYXNzaWduKFxuICAgICAgICAgICAge30sXG4gICAgICAgICAgICBERUZBVUxUX1NFVFRJTkdTLFxuICAgICAgICAgICAgdXNlclNldHRpbmdzID8gdXNlclNldHRpbmdzIDogTkVXX1VTRVJfU0VUVElOR1NcbiAgICAgICAgKTtcbiAgICB9XG5cbiAgICBhc3luYyBzYXZlU2V0dGluZ3MoKSB7XG4gICAgICAgIGF3YWl0IHRoaXMuc2F2ZURhdGEodGhpcy5zZXR0aW5ncyk7XG4gICAgfVxufVxuIl0sIm5hbWVzIjpbIlN1Z2dlc3RNb2RhbCIsIlNldHRpbmciLCJQbHVnaW5TZXR0aW5nVGFiIiwiUGxhdGZvcm0iLCJQbHVnaW4iXSwibWFwcGluZ3MiOiI7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLElBQUksYUFBYSxHQUFHLFNBQVMsQ0FBQyxFQUFFLENBQUMsRUFBRTtBQUNuQyxJQUFJLGFBQWEsR0FBRyxNQUFNLENBQUMsY0FBYztBQUN6QyxTQUFTLEVBQUUsU0FBUyxFQUFFLEVBQUUsRUFBRSxZQUFZLEtBQUssSUFBSSxVQUFVLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsU0FBUyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDcEYsUUFBUSxVQUFVLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxLQUFLLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxJQUFJLE1BQU0sQ0FBQyxTQUFTLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDMUcsSUFBSSxPQUFPLGFBQWEsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7QUFDL0IsQ0FBQyxDQUFDO0FBQ0Y7QUFDTyxTQUFTLFNBQVMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFO0FBQ2hDLElBQUksSUFBSSxPQUFPLENBQUMsS0FBSyxVQUFVLElBQUksQ0FBQyxLQUFLLElBQUk7QUFDN0MsUUFBUSxNQUFNLElBQUksU0FBUyxDQUFDLHNCQUFzQixHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRywrQkFBK0IsQ0FBQyxDQUFDO0FBQ2xHLElBQUksYUFBYSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztBQUN4QixJQUFJLFNBQVMsRUFBRSxHQUFHLEVBQUUsSUFBSSxDQUFDLFdBQVcsR0FBRyxDQUFDLENBQUMsRUFBRTtBQUMzQyxJQUFJLENBQUMsQ0FBQyxTQUFTLEdBQUcsQ0FBQyxLQUFLLElBQUksR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxTQUFTLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUM7QUFDekYsQ0FBQztBQW9GRDtBQUNPLFNBQVMsU0FBUyxDQUFDLE9BQU8sRUFBRSxVQUFVLEVBQUUsQ0FBQyxFQUFFLFNBQVMsRUFBRTtBQUM3RCxJQUFJLFNBQVMsS0FBSyxDQUFDLEtBQUssRUFBRSxFQUFFLE9BQU8sS0FBSyxZQUFZLENBQUMsR0FBRyxLQUFLLEdBQUcsSUFBSSxDQUFDLENBQUMsVUFBVSxPQUFPLEVBQUUsRUFBRSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRTtBQUNoSCxJQUFJLE9BQU8sS0FBSyxDQUFDLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQyxFQUFFLFVBQVUsT0FBTyxFQUFFLE1BQU0sRUFBRTtBQUMvRCxRQUFRLFNBQVMsU0FBUyxDQUFDLEtBQUssRUFBRSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUU7QUFDbkcsUUFBUSxTQUFTLFFBQVEsQ0FBQyxLQUFLLEVBQUUsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLFNBQVMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUU7QUFDdEcsUUFBUSxTQUFTLElBQUksQ0FBQyxNQUFNLEVBQUUsRUFBRSxNQUFNLENBQUMsSUFBSSxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLEdBQUcsS0FBSyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxFQUFFLFFBQVEsQ0FBQyxDQUFDLEVBQUU7QUFDdEgsUUFBUSxJQUFJLENBQUMsQ0FBQyxTQUFTLEdBQUcsU0FBUyxDQUFDLEtBQUssQ0FBQyxPQUFPLEVBQUUsVUFBVSxJQUFJLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBRSxDQUFDLENBQUM7QUFDOUUsS0FBSyxDQUFDLENBQUM7QUFDUCxDQUFDO0FBQ0Q7QUFDTyxTQUFTLFdBQVcsQ0FBQyxPQUFPLEVBQUUsSUFBSSxFQUFFO0FBQzNDLElBQUksSUFBSSxDQUFDLEdBQUcsRUFBRSxLQUFLLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBRSxXQUFXLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxFQUFFLEVBQUUsR0FBRyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQztBQUNySCxJQUFJLE9BQU8sQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxPQUFPLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLFFBQVEsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxPQUFPLE1BQU0sS0FBSyxVQUFVLEtBQUssQ0FBQyxDQUFDLE1BQU0sQ0FBQyxRQUFRLENBQUMsR0FBRyxXQUFXLEVBQUUsT0FBTyxJQUFJLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDO0FBQzdKLElBQUksU0FBUyxJQUFJLENBQUMsQ0FBQyxFQUFFLEVBQUUsT0FBTyxVQUFVLENBQUMsRUFBRSxFQUFFLE9BQU8sSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUU7QUFDdEUsSUFBSSxTQUFTLElBQUksQ0FBQyxFQUFFLEVBQUU7QUFDdEIsUUFBUSxJQUFJLENBQUMsRUFBRSxNQUFNLElBQUksU0FBUyxDQUFDLGlDQUFpQyxDQUFDLENBQUM7QUFDdEUsUUFBUSxPQUFPLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsSUFBSTtBQUN0RCxZQUFZLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0FBQ3pLLFlBQVksSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztBQUNwRCxZQUFZLFFBQVEsRUFBRSxDQUFDLENBQUMsQ0FBQztBQUN6QixnQkFBZ0IsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsTUFBTTtBQUM5QyxnQkFBZ0IsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsT0FBTyxFQUFFLEtBQUssRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxFQUFFLEtBQUssRUFBRSxDQUFDO0FBQ3hFLGdCQUFnQixLQUFLLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTO0FBQ2pFLGdCQUFnQixLQUFLLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxTQUFTO0FBQ2pFLGdCQUFnQjtBQUNoQixvQkFBb0IsSUFBSSxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLFNBQVMsRUFBRTtBQUNoSSxvQkFBb0IsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsS0FBSyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sRUFBRTtBQUMxRyxvQkFBb0IsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsTUFBTSxFQUFFO0FBQ3pGLG9CQUFvQixJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxNQUFNLEVBQUU7QUFDdkYsb0JBQW9CLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUM7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
