#!/bin/bash
# queue.sh -- serial job queue: ONE job at a time, niceness 10, OpenFOAM v2606 environment.
# Jobs are shell scripts dropped into queue/pending/ (run in name order); output goes to
# queue/logs/<job>.log; finished jobs move to queue/done/.  Stop with: touch queue/STOP
Q="$(cd "$(dirname "$0")/.." && pwd)/queue"
echo "queue runner started $(date -Is) pid $$" >> "$Q/runner.log"
while [ ! -e "$Q/STOP" ]; do
    job=$(ls "$Q/pending" 2>/dev/null | sort | head -n 1)
    if [ -z "$job" ]; then sleep 5; continue; fi
    echo "$(date -Is) start $job" >> "$Q/runner.log"
    mv "$Q/pending/$job" "$Q/running_$job"
    nice -n 10 openfoam2606 -c "bash '$Q/running_$job'" > "$Q/logs/$job.log" 2>&1
    echo "$(date -Is) end   $job rc=$?" >> "$Q/runner.log"
    mv "$Q/running_$job" "$Q/done/$job"
done
echo "queue runner stopped $(date -Is)" >> "$Q/runner.log"
