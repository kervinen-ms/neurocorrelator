from pathlib import Path

import fire

from src.neurocorrelator.qatm_pytorch import run_pipeline


def main(
    cuda: bool = True,
    sample_image: str = "samples/image.png",
    template_images_dir: str = "templates/",
    result_images_dir: str = "result/",
    alpha: float = 25,
    thresh_csv: str = "thresh_template.csv",
    use_trained: bool = True,
):
    """
    QATM Pytorch Inference

    Args:
        cuda: Использовать CUDA (по умолчанию: False)
        sample_image: Путь к тестовому изображению (по умолчанию: "neurocorrelator/result/sample/sample1.jpg")
        template_images_dir: Директория с шаблонами (по умолчанию: "neurocorrelator/template/")
        result_images_dir: Директория для сохранения результатов (по умолчанию: "neurocorrelator/result/")
        alpha: Параметр alpha (по умолчанию: 25)
        thresh_csv: CSV-файл с порогами (по умолчанию: "thresh_template.csv")
    """
    sample_path = Path.cwd() / sample_image
    template_path = Path.cwd() / template_images_dir

    result_path = Path(result_images_dir).resolve()
    result_path.mkdir(parents=True, exist_ok=True)

    thresh_csv_path = Path(thresh_csv).resolve()

    _ = run_pipeline(
        image_path=sample_path,
        template_dir=template_path,
        result_dir=result_path,
        alpha=alpha,
        use_cuda=cuda,
        use_trained=use_trained,
        threshold_csv=thresh_csv_path,
    )


if __name__ == "__main__":
    fire.Fire(main)
