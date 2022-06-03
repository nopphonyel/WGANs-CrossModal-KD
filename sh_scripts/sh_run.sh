# Do an arguments recieve
while getopts m:e:p: flag; do
  case "${flag}" in
  m) MODE=${OPTARG} ;;
  e) EXP=${OPTARG} ;;
  p) PT_PATH=${OPTARG} ;;
  *) echo "<?> Unknown arg: ${flag}" ;;
  esac
done

# Set the root directory of the project
#export PYTHONPATH="/root/MountPoint/CrossModal_KD"
if [ -z "$PT_PATH" ]; then
  export PYTHONPATH="$(pwd)"
else
  export PYTHONPATH="$PT_PATH"
fi
#export

# python "/home/bci/working_dir/WGAN-ImageRecon/stuff_of_test/j1_loss_test.py"
# python "/home/bci/working_dir/WGAN-ImageRecon/stuff_of_test/merge_fe_test.py";
# python "/home/bci/working_dir/WGAN-ImageRecon/stuff_of_test/merge_fe_f1_score.py"
# python "/home/bci/working_dir/WGAN-ImageRecon/train_script/merge_fe_no_conditioned_label.py"

echo "MODE: $MODE"
echo "EXP: $EXP"
echo "PYTHONPATH: $PYTHONPATH"
python3 "$PYTHONPATH/exp/$MODE/$EXP/exec_me.py"
