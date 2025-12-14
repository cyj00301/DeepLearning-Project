import os
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from model import Model
from data_loader import RawDataLoader
from utils import DATA_MODALITIES, GDSC_RAW_DATA_FOLDER, GDSC_SCREENING_DATA_FOLDER

AE_LATENT_DIM = 50

app = FastAPI(title="Drug Response Prediction")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None
cell_tensor = None
drug_tensor = None
cell_names = []
drug_names = []
cell_index = {}
drug_index = {}


def _build_feature_frames(data_dict):
    cell_types = sorted([k for k in data_dict if k.startswith("cell")])
    drug_types = sorted([k for k in data_dict if k.startswith("drug")])

    cell_sizes = [data_dict[t].shape[1] for t in cell_types]
    drug_sizes = [data_dict[t].shape[1] for t in drug_types]

    cell_df = pd.concat([data_dict[t].add_suffix(f"_{t}") for t in cell_types], axis=1)
    drug_df = pd.concat([data_dict[t].add_suffix(f"_{t}") for t in drug_types], axis=1)
    return cell_df, drug_df, cell_sizes, drug_sizes


def _find_weight_path():
    candidates = [
        "MODEL.pth",
        os.path.join("data", "MODEL.pth"),
        os.path.join("data", "model", "MODEL.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("모델 가중치 파일을 찾을 수 없습니다.")


def _load_state_dict_safely(model, state_dict):
    model_keys = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_keys and model_keys[k].shape == v.shape:
            filtered[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    return missing_keys, unexpected_keys


@app.on_event("startup")
def load_artifacts():
    global model, cell_tensor, drug_tensor, cell_names, drug_names, cell_index, drug_index

    data_dict, _ = RawDataLoader.load_data(
        data_modalities=DATA_MODALITIES,
        raw_file_directory=GDSC_RAW_DATA_FOLDER,
        screen_file_directory=GDSC_SCREENING_DATA_FOLDER,
        sep="\t",
    )

    cell_df, drug_df, cell_sizes, drug_sizes = _build_feature_frames(data_dict)
    cell_names = cell_df.index.tolist()
    drug_names = drug_df.index.tolist()
    cell_index = {name: idx for idx, name in enumerate(cell_names)}
    drug_index = {name: idx for idx, name in enumerate(drug_names)}

    cell_tensor = F.normalize(torch.tensor(cell_df.values, dtype=torch.float32), dim=0)
    drug_tensor = F.normalize(torch.tensor(drug_df.values, dtype=torch.float32), dim=0)

    weight_path = _find_weight_path()
    model = Model(cell_sizes, drug_sizes, AE_LATENT_DIM, AE_LATENT_DIM)

    state_dict = torch.load(weight_path, map_location=device)
    missing, unexpected = _load_state_dict_safely(model, state_dict)
    if missing:
        print(f"[정보] 체크포인트에 없어서 초기화된 파라미터: {missing}")
    if unexpected:
        print(f"[정보] 현재 모델에 없는 체크포인트 파라미터: {unexpected}")

    model.to(device)
    model.eval()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "cell_names": cell_names, "drug_names": drug_names},
    )


@app.post("/predict", response_class=JSONResponse)
async def predict_json(cell_name: str = Form(...), drug_name: str = Form(...)):
    if cell_name not in cell_index or drug_name not in drug_index:
        return JSONResponse(
            status_code=400,
            content={"error": "선택한 세포주 또는 약물이 데이터에 없습니다."},
        )

    cell_feat = cell_tensor[cell_index[cell_name]].unsqueeze(0).to(device)
    drug_feat = drug_tensor[drug_index[drug_name]].unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, pred = model(cell_feat, drug_feat)
        prob = float(pred.squeeze().item())

    is_sensitive = prob >= 0.5

    result = {
        "cell": cell_name,
        "drug": drug_name,
        "sensitive_prob": round(prob, 4),
        "resistance_prob": round(1 - prob, 4),
        "prediction": "민감 (Sensitive)" if is_sensitive else "내성 (Resistant)",
        "threshold": 0.5,
    }

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
