# Drosobot
A query engine for Drosophila datasets.

## Installation

* Download the default provided dataset [here](https://drive.google.com/drive/folders/1C4Rq5-3qGLZ2ODWD13a5JZxATA2QHpj_?usp=sharing).
* Run the service using Flask:
```bash
export FLASK_APP=[MY_PATH]/drosobot_app
srun flask run --host=0.0.0.0 --port=[INSERT PORT NUMBER]
```