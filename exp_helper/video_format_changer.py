import subprocess
import os

FOLDER_LIST = [
    # # paper_nico
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/nico_trackings/hiob-execution-wtmpc161-2019-09-04-19.50.22.057631/images/push_blue_ball_01",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/nico_trackings/hiob-execution-wtmpc161-2019-09-04-19.50.22.057631/images/scoot_green_car_01",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/nico_trackings/hiob-execution-wtmpc161-2019-09-04-19.50.22.057631/images/shake_big_light_banana_01",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/nico_trackings/hiob-execution-wtmpc161-2019-09-04-19.50.22.057631/images/shake_small_banana_06",
    # # paper_emil
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/emil_trackings/hiob-execution-wtmpc30-2019-09-04-14.59.51.848507/images/lift_soft-banana_id-1-100_video",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/emil_trackings/hiob-execution-wtmpc30-2019-09-04-14.59.51.848507/images/pull_yellow-car_id-1-73_video",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/emil_trackings/hiob-execution-wtmpc30-2019-09-04-14.59.51.848507/images/push_soft-red-ball_id-1-45_video",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/emil_trackings/hiob-execution-wtmpc30-2019-09-04-14.59.51.848507/images/scoot_heavy-green-car_id-0-17_video",
    # # paper tb100
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/BlurBody",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/Football",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/Lemming",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/MotorRolling",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/paper_hiob/Walking",
    # # se cand_stat_cont
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/se_cand_stat_cont/images/Biker",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/se_cand_stat_cont/images/Car4",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/se_cand_stat_cont/images/CarScale",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/se_cand_stat_cont/images/David",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/se_cand_stat_cont/images/Lemming",
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/se_cand_stat_cont/images/Walking",
    # # se_dsst_stat_cont
    # "/informatik3/wtm/scratch/Heinrich/draft-videos/HIOB/se_dsst_stat_cont/images/push_soft-red-ball_id-1-45_video"
]

for img_folder in FOLDER_LIST:
    print("converting frames.mkv and heatmaps.mkv in {}".format(img_folder))

    cmd_frames = ("ffmpeg -i " + os.path.join(img_folder, "frames.mkv") +
                  " -c:v libx264 -crf 0 -preset ultrafast -c:a libmp3lame -b:a 320k -vf format=yuv420p " + os.path.join(
                img_folder, "frames_yuv420p.mp4"))

    cmd_heatmaps = ("ffmpeg -i " + os.path.join(img_folder, "heatmaps.mkv") +
                    " -c:v libx264 -crf 0 -preset ultrafast -c:a libmp3lame -b:a 320k -vf format=yuv420p " + os.path.join(
                img_folder,
                "heatmaps_yuv420p.mp4"))

    subprocess.call(cmd_frames, shell=True)
    subprocess.call(cmd_heatmaps, shell=True)

    # p = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE, shell=True)
    # print(p.communicate())
