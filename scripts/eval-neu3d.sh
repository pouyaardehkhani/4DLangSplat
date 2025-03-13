########## exp setup ##########
clip_feat_dim=3
dataset_name=coffee_martini


########## Evaluate ##########
cd eval
python eval.py --dataset_type neu3d \
    --annotation_folder ../data/neu3d/${dataset_name}/annotations \
    --exp_name ${dataset_name}/${dataset_name} \
    --feat_dim ${clip_feat_dim} \
    --iterations 10000 \
    --ae_ckpt_path ../output/neu3d/${dataset_name}/clip_best_ckpt.pth \
    --decoder_hidden_dims 16 32 64 128 256 256 512

### For more detailed results, uncomment the following lines ###
    # --visualize_results \
    # --detail_results