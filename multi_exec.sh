# List of mode and experiment to run in parallel
mode=("whole" "whole")
exp=("exp11" "exp12")

# This will loop through all the arrays of mode and exp. Each iteration will create a screen session for each
# experiment session.
for ((i = 0; i < ${#mode[@]}; i++)); do
  EXEC_STR="echo \"Session \"${mode[$i]}:${exp[$i]}\"\";"
  EXEC_STR="$EXEC_STR ./sh_scripts/sh_run.sh -m \"${mode[$i]}\" -e \"${exp[$i]}\";"
  EXEC_STR="$EXEC_STR exec bash;"
  session_name="${mode[$i]}.${exp[$i]}"
  screen -dmS "$session_name" bash -c "$EXEC_STR"
  screen -r "$session_name"
done
