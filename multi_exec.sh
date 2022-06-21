# List of mode and experiment to run in parallel
mode=("whole")
exp=("exp10")

# This will loop through all the arrays of mode and exp. Each iteration will create a screen session for each
# experiment session.
for ((i = 0; i < ${#mode[@]}; i++)); do
  session_name="${mode[$i]}-${exp[$i]}"
  session_name=${session_name//\//-} # Replace '/' to '-'

  EXEC_STR="echo \"Session \"$session_name\"\";"
  EXEC_STR="$EXEC_STR ./sh_scripts/sh_run.sh -m \"${mode[$i]}\" -e \"${exp[$i]}\";"
  EXEC_STR="$EXEC_STR exec bash;"

  screen -dmS "$session_name" bash -c "$EXEC_STR"
  screen -r "$session_name"
done
