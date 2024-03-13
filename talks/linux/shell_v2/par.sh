for i in {1..5}; do
    echo "Starting $i" && sleep 5 && echo "Ending $i" &
done
wait
echo "DONE"
