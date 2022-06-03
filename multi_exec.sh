# List of mode and experiment to run in parallel
mode=("fe" "fe" "fe")
exp=("alex_fid" "exp01" "exp01_1")

# This will loop through all the arrays of mode and exp. Each iteration will create a screen session for each
# experiment session.
for ((i = 0; i < ${#mode[@]}; i++)); do
  EXEC_STR="echo \"Session \"${mode[$i]}:${exp[$i]}\"\";"
  EXEC_STR="$EXEC_STR ./sh_scripts/sh_run.sh -m \"${mode[$i]}\" -e \"${exp[$i]}\";"
  EXEC_STR="$EXEC_STR exec bash;"
  screen -dmS "sess$i" bash -c "$EXEC_STR"
  screen -r "sess$i"
done
