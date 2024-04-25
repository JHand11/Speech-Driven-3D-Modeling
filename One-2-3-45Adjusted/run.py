import os
import torch
import argparse
import shlex
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer, predict_rotation_gradio, infer_shifted_images_for_folder
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev


def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def stage1_run(model, device, exp_dir,
               input_im, supp_imgs, scale, ddim_steps):
    # folder to save the stage 1 images
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input

    output_ims = []
    output_ims_2 = []
    for i, img in enumerate(supp_imgs):
        # Check if img is a path or an already opened image
        if isinstance(img, str):  # img is a path
            img = Image.open(img).convert("RGB")
        elif not img.mode == "RGB":  # img is an opened image but not in RGB mode
            img = img.convert("RGBA")
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        white_bg = Image.new('RGBA', img.size, (255,255,255,255))
        white_bg.paste(img, mask=img)
        white_bg = white_bg.convert('RGB')
        # Assuming img is now an opened Image object in RGB mode
        white_bg.save(os.path.join(stage1_dir, f"{i}.png"))
        output_ims.append(white_bg)

    # stage 2 for the first image
    # infer 4 nearby views for an image to estimate the polar angle of the input
    stage2_steps = 50 # ddim_steps
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
    # estimate the camera pose (elevation) of the input image.
    try:
        polar_angle = int(estimate_elev(exp_dir))
    except:
        print("Failed to estimate polar angle")
        polar_angle = 90
    print("Estimated polar angle:", polar_angle)
    gen_poses(exp_dir, polar_angle)

    # stage 1: generate another 4 views at a different elevation
    delta_x = 30 if polar_angle <= 75 else -30
    pic_start = 4 if polar_angle <= 75 else 8
    delta_y = 30  # delta_y is always 30


    infer_shifted_images_for_folder(
        model=model,
        input_dir_path = stage1_dir,
        save_path_stage2 = stage1_dir,
        delta_x=delta_x,
        delta_y=delta_y,
        device=device,
        pic_start=pic_start,  # controls file naming
        ddim_steps=75,
        scale=3.0
    )

    torch.cuda.empty_cache()
    # Adjust the return value to match your specific needs
    return 90 - polar_angle

def stage2_run(model, device, exp_dir,
               elev, scale, stage2_steps=50):
    # stage 2 for the remaining 7 images, generate 7*4=28 views
    if 90-elev <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct(exp_dir, output_format=".ply", device_idx=0, resolution=512):
    exp_dir = os.path.abspath(exp_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')
    
    exp_dir_escaped = shlex.quote(exp_dir)
    device_idx_str = str(device_idx)
    resolution_str = str(resolution)


    bash_script = f'CUDA_VISIBLE_DEVICES={device_idx_str} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {exp_dir_escaped} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf \
                    --resolution {resolution_str}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(exp_dir, f"mesh.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(exp_dir, output_format=output_format)


def predict_multiview(shape_dir, args):
    device = f"cuda:{args.gpu_idx}"

    # initialize the zero123 model
    models = init_model(device, '/home/vislab-002/local/HandAntesMQP/zero123-xl.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]

    # initialize the Segment Anything model
    predictor = sam_init(args.gpu_idx)
    input_raw = Image.open(args.img_path)

    # preprocess the input image
    input_256 = preprocess(predictor, input_raw)

    supp_imgs = [Image.open(img_path) for img_path in [args.supp_img1, args.supp_img2, args.supp_img3, args.supp_img4]]
    # generate multi-view images in two stages with Zero123.
    # first stage: generate N=8 views cover 360 degree of the input shape.
    elev = stage1_run(model_zero123, device, shape_dir, input_256, supp_imgs, scale=3, ddim_steps=75)
    # second stage: 4 local views for each of the first-stage view, resulting in N*4=32 source view images.
    stage2_run(model_zero123, device, shape_dir, elev, scale=3, stage2_steps=75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_path', type=str, default="./demo/demo_examples/01_wild_hydrant.png", help='Path to the input image')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision')
    parser.add_argument('--mesh_resolution', type=int, default=256, help='Mesh resolution')
    parser.add_argument('--output_format', type=str, default=".ply", help='Output format: .ply, .obj, .glb')
    parser.add_argument('--supp_img1', type=str, required=True, help='Path to the supplementary image 1')
    parser.add_argument('--supp_img2', type=str, required=True, help='Path to the supplementary image 2')
    parser.add_argument('--supp_img3', type=str, required=True, help='Path to the supplementary image 3')
    parser.add_argument('--supp_img4', type=str, required=True, help='Path to the supplementary image 4')

    args = parser.parse_args()

    assert(torch.cuda.is_available())

    shape_id = args.img_path.split('/')[-1].split('.')[0]
    shape_dir = f"./exp/{shape_id}"
    os.makedirs(shape_dir, exist_ok=True)

    predict_multiview(shape_dir, args)

    # utilize cost volume-based 3D reconstruction to generate textured 3D mesh
    mesh_path = reconstruct(shape_dir, output_format=args.output_format, device_idx=args.gpu_idx, resolution=args.mesh_resolution)
    print("Mesh saved to:", mesh_path)
