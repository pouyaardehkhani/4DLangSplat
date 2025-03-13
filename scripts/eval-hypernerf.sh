########## exp setup ##########
clip_feat_dim=3
video_feat_dim=6
dataset_name=americano # chickchicken, espresso, split-cookie

########## Evaluate ##########
cd eval
python eval.py --dataset_type hypernerf \
    --annotation_folder ../data/hypernerf/${dataset_name}/annotations \
    --exp_name ${dataset_name}/${dataset_name} \
    --feat_dim ${clip_feat_dim} \
    --video_feat_dim ${video_feat_dim} \
    --iterations 10000 \
    --video_eval_iterations 20000 \
    --ae_ckpt_path ../output/hypernerf/${dataset_name}/clip_best_ckpt.pth \
    --video_ae_ckpt_path ../output/hypernerf/${dataset_name}/video_best_ckpt.pth \
    --video_feat_dir ${dataset_name}/${dataset_name} \
    --apply_video_search \
    --smooth_feature_post

### For more detailed results, uncomment the following lines ###
    # --visualize_results \
    # --detail_results