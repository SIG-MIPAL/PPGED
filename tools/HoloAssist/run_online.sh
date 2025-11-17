python train.py configs/HoloAssist/holo_bgr1.0_verb_b1_n2_smooth_causal_gmm_dmap.yaml --output final >> log/holoverb_online/train.log
python test_ed.py configs/HoloAssist/holo_bgr1.0_verb_b1_n2_smooth_causal_gmm_dmap.yaml ckpt/HoloAssist_taskgraph_dmap/holo_bgr1.0_verb_b1_n2_smooth_causal_gmm_dmap_final >> log/holoverb_online/test_ed.log


python train.py configs/HoloAssist/holo_bgr1.0_noun_b1_n2_smooth_causal_gmm_dmap.yaml --output final >> log/holonoun_online/train.log
python test_ed.py configs/HoloAssist/holo_bgr1.0_noun_b1_n2_smooth_causal_gmm_dmap.yaml ckpt/HoloAssist_taskgraph_dmap/holo_bgr1.0_noun_b1_n2_smooth_causal_gmm_dmap_final >> log/holonoun_online/test_ed.log


python metric_vis_holo.py --dirs HoloAssist_taskgraph_dmap --dirname holo_bgr1.0_verb_b1_n2_smooth_causal_gmm_dmap_final --action verb -ed --online --threshold -100.0 >> log/holoverb_online/metric.log
