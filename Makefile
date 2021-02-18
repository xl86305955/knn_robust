All:
	python run_experiment.py mnist nn DEEPFOOL

halfmoon_fgsm:
	python run_experiment.py halfmoon nn FGSM
	python plot.py halfmoon FGSM

halfmoon_deepfool:
	python run_experiment.py halfmoon nn DEEPFOOL
	python plot.py halfmoon DEEPFOOL

halfmoon_cw:
	python run_experiment.py halfmoon nn CW
	python plot.py halfmoon CW

abalone_fgsm:
	python run_experiment.py abalone nn FGSM
	python plot.py abalone FGSM

abalone_deepfool:
	python run_experiment.py abalone nn DEEPFOOL
	python plot.py abalone DEEPFOOL

abalone_cw:
	python run_experiment.py abalone nn CW
	python plot.py abalone CW

mnist_fgsm:
	python run_experiment.py mnist nn FGSM
	python plot.py mnist FGSM

mnist_deepfool:
	python run_experiment.py mnist nn DEEPFOOL
	python plot.py mnist DEEPFOOL

mnist_cw:
	python run_experiment.py mnist nn CW
	python plot.py mnist CW
