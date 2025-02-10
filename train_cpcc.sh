export CUDA_VISIBLE_DEVICES=0


python metadrive_RLtrain.py  \
    --use_cpql_cl \
    --seed 0 \
    --num_classes 21 \
    --v_max 2 \
    --v_expand_mode "both" \
    --v_expand 0 \
    --alpha 0.1
   
