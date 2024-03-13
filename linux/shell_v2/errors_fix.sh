cat /tmp/this_file_does_not_exist 2>/dev/null || echo "The file does not exist. Exiting now" && exit 1
touch /tmp/this_file_does_not_exist/some/dir/file  2>/dev/null || echo "Some problem occurred. Exiting now" && exit 1
echo "The above is a result of an operation"
