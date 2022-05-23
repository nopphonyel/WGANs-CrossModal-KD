# Set the root directory of the project
EXP='exp01'
export PYTHONPATH="/root/MountPoint/CrossModal_KD";

# python "/home/bci/working_dir/WGAN-ImageRecon/stuff_of_test/j1_loss_test.py"
# python "/home/bci/working_dir/WGAN-ImageRecon/stuff_of_test/merge_fe_test.py";
# python "/home/bci/working_dir/WGAN-ImageRecon/stuff_of_test/merge_fe_f1_score.py"
# python "/home/bci/working_dir/WGAN-ImageRecon/train_script/merge_fe_no_conditioned_label.py"
python3 "/root/MountPoint/CrossModal_KD/exp/$EXP/exec_me.py";
