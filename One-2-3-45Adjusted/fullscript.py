#!/home/vislab-002/anaconda3/envs/One2345/bin/python
import os
import subprocess
import argparse
from datetime import datetime
import paramiko
from scp import SCPClient
import shutil

def create_scp_client(host, port, username, password):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)
    return SCPClient(ssh.get_transport())

def transfer_directory_with_paramiko(source_path, destination_path, host, username, password):
    scp = create_scp_client(host, 22, username, password)
    try:
        scp.put(source_path, recursive=True, remote_path=destination_path)
        print("Directory successfully transferred.")
    except Exception as e:
        print(f"Failed to transfer directory: {e}")
    finally:
        scp.close()

def create_unique_directory(base_path):
    """Creates a unique directory based on the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = os.path.join(base_path, f"run_{timestamp}")
    os.makedirs(directory, exist_ok=True)
    return directory


def get_recent_files(directory, pattern, count):
    command = f'find "{directory}" -maxdepth 1 -name "{pattern}" -printf "%T@ %p\\n" | sort -n | tail -n {count}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print("Error:", stderr.decode('utf-8'))
        return []
    
    files = [line.split(' ', 1)[1].strip() for line in stdout.decode('utf-8').splitlines()]
    return files

def split_images_function(images, output_dir):
    for image in images:
        os.makedirs(output_dir, exist_ok=True)
        
        split_command =(        
            f'cd ~/local/HandAntesMQP; '
            f'source Hand/bin/activate; ' 
            f'"/media/vislab-002/SP2 4TB/OpenLRM/split_image.py" "{image}" "{output_dir}"'
        )
        process = subprocess.Popen(split_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error splitting image: {image}", stderr.decode('utf-8'))
        else:
            print(f"Image split successfully: {image}")

def process_split_images(images, command_base_dir, resolution, unique_dir):
    if len(images) != 4:
        print("Error: Exactly four images are required.")
        return
    python_script_path = os.path.join(command_base_dir, 'run.py')
    python_executable = "/home/vislab-002/anaconda3/envs/One2345/bin/python"
    command = (
        f'{python_executable} "{python_script_path}" '
        f'--img_path "{images[0]}" '
        f'--supp_img1 "{images[0]}" '
        f'--supp_img2 "{images[1]}" '
        f'--supp_img3 "{images[2]}" '
        f'--supp_img4 "{images[3]}" '
        '--half_precision '
        f'--mesh_resolution {resolution}'
    )

    process = subprocess.Popen(command, shell=True, cwd=command_base_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("Error processing images:", stderr.decode('utf-8'))
    else:
        print("Images processed successfully.")
        # Parse stdout for mesh file path
        output = stdout.decode('utf-8')
        mesh_file_path = None
        for line in output.splitlines():
            if line.startswith("Mesh saved to:"):
                mesh_file_path = line.split(":")[1].strip()
                break

        if mesh_file_path:
            print(f"Mesh file path: {mesh_file_path}")
            # Copy the mesh file to the unique directory
            shutil.copy(mesh_file_path, unique_dir)
            print(f"Mesh file copied to {unique_dir}")

            # Determine the directory of the mesh file and locate the stage1_8 directory
            mesh_dir_path = os.path.dirname(mesh_file_path)
            stage1_8_dir = os.path.join(mesh_dir_path, "stage1_8")
            if os.path.exists(stage1_8_dir):
                for file in os.listdir(stage1_8_dir):
                    if file.endswith(".png"):
                        src_file_path = os.path.join(stage1_8_dir, file)
                        shutil.copy(src_file_path, unique_dir)
                        print(f"Copied {file} to {unique_dir}")
            else:
                print("stage1_8 directory does not exist.")
        else:
            print("Mesh file path not found in output.")

        return output

def main(args):

    unique_dir = create_unique_directory(r'/media/vislab-002/SP2 4TB/One-2-3-45/Uniques')
    generate_image_command = (
        f'cd ~/local/HandAntesMQP; '
        f'source Hand/bin/activate; '
        f'cd kohya_ss; '
        f'python3 sdxl_gen_img.py '
        f'--ckpt "/home/vislab-002/local/HandAntesMQP/32724SDXL.ckpt" '
        f'--outdir "{unique_dir}" '
        f'--xformers --bf16 --W 1024 --H 1024 --scale 12.5 --sampler k_euler_a --steps 256 '
        f'--batch_size 8 --images_per_prompt 1 --prompt "{args.prompt}"'
    )
    subprocess.run(generate_image_command, shell=True, executable='/bin/bash', check=True)

    recent_images = get_recent_files(unique_dir, "*.png", 1)
    print("Recent Image:", recent_images)

    split_images_function(recent_images, unique_dir)

    split_images = get_recent_files(unique_dir, "*.png", 4)

    One2345dir = r'/media/vislab-002/SP2 4TB/One-2-3-45'

    output = process_split_images(split_images, One2345dir, args.mesh_resolution, unique_dir)
    print(output)
    transfer_directory_with_paramiko(unique_dir, 'C:/Users/handj/Documents', '130.215.219.48', 'handj', 'Calripken#12')
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images based on a prompt.')
    parser.add_argument('prompt', type=str, help='A prompt for image generation')
    parser.add_argument('mesh_resolution', type=int, help='The resolution of the exported mesh')  # Changed type to int
    args = parser.parse_args()
    
    main(args)
