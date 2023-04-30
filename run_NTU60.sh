date
#SkeAttnCLR
#joint
python main.py pretrain_SkeAttnLWL --config config/SkeAttnCLR/NTU60/joint/pretrain_xsub.yaml
python main.py linear_evaluation_LWL --config config/SkeAttnCLR/NTU60/joint/linear_eval_xsub.yaml

python main.py pretrain_SkeAttnLWL --config config/SkeAttnCLR/NTU60/joint/pretrain_xview.yaml
python main.py linear_evaluation_LWL --config config/SkeAttnCLR/NTU60/joint/linear_eval_xview.yaml

#motion
python main.py pretrain_SkeAttnLWL --config config/SkeAttnCLR/NTU60/motion/pretrain_xsub.yaml
python main.py linear_evaluation_LWL --config config/SkeAttnCLR/NTU60/motion/linear_eval_xsub.yaml

python main.py pretrain_SkeAttnLWL --config config/SkeAttnCLR/NTU60/motion/pretrain_xview.yaml
python main.py linear_evaluation_LWL --config config/SkeAttnCLR/NTU60/motion/linear_eval_xview.yaml

#bone
python main.py pretrain_SkeAttnLWL --config config/SkeAttnCLR/NTU60/bone/pretrain_xsub.yaml
python main.py linear_evaluation_LWL --config config/SkeAttnCLR/NTU60/bone/linear_eval_xsub.yaml

python main.py pretrain_SkeAttnLWL --config config/SkeAttnCLR/NTU60/bone/pretrain_xview.yaml
python main.py linear_evaluation_LWL --config config/SkeAttnCLR/NTU60/bone/linear_eval_xview.yaml
