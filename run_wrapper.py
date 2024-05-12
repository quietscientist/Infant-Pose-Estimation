import subprocess
import time

def run_command():
    cmd = [
        'python', '/workspaces/Infant-Pose-Estimation/syn_generation/smplifyx/main_from_kp.py',
        '--config', '/workspaces/Infant-Pose-Estimation/syn_generation/cfg_files/fit_smil.yaml',
        '--data_folder', '/workspaces/Infant-Pose-Estimation/syn_generation/data',
        '--output_folder', '/workspaces/Infant-Pose-Estimation/syn_generation/data/output',
        '--visualize', 'False',
        '--model_folder', '/workspaces/Infant-Pose-Estimation/syn_generation/models',
        '--vposer_ckpt', '/workspaces/Infant-Pose-Estimation/syn_generation/vposer_checkpoint',
        '--part_segm_fn', 'smplx_parts_segm.pkl'
    ]

    while True:
        try:
            # Run the command and wait for it to complete
            subprocess.check_call(cmd)
            print("Command executed successfully.")
            break  # Exit the loop if the command was successful
        except subprocess.CalledProcessError:
            print("Error encountered. Rerunning the command...")
            time.sleep(2)  # Wait for 2 seconds before retrying

if __name__ == '__main__':
    run_command()
