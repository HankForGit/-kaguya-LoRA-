# -kaguya-LoRA-
json文件为微调所需训练集以及原始字幕文件
其余为训练文件，均使用CC+ds搭建

bullet
•	Fine-tuned Qwen 3.5-4B with LoRA to imitate Kaguya’s speaking style, building a full pipeline from data collection and cleaning to training and inference.
•	Replaced inefficient OCR-based subtitle extraction with a subtitle-file parsing workflow, converting raw subtitles into structured JSON and separating character dialogue into input/output pairs.
•	Performed manual data validation to improve training quality, then iteratively tuned hyperparameters based on model behavior and community recommendations.
•	Added timestamped checkpoints and a custom inference script to compare model versions quickly, preventing adapter/version mismatch during evaluation.
•	Deployed the final model in LM Studio and used system prompts to stabilize output style, producing a demo that received positive feedback from peers.
