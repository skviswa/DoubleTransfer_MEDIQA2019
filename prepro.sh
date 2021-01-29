mkdir -p ../model_data/
mkdir -p ../mediqa_processed/
mkdir -p ../data/mediqa/task1_mednli/
mkdir -p ../data/mediqa/task2_rqe/
mkdir -p ../data/mediqa/MedQuAD/
mkdir -p ../data/mediqa/task3_qa/

ROOT_DIR=$1
# preprocessing, uncased
python prepro_mediqa.py --root_dir "$ROOT_DIR"
# preprocessing, cased
python prepro_mediqa.py --root_dir "$ROOT_DIR" --cased
# preprocessing, uncased, with SciBERT vocabulary
python prepro_mediqa.py --root_dir "$ROOT_DIR" --sci_vocab
# preeprocessing, uncased, with BlueBERT vocabulary
python prepro_mediqa.py --bluebert_vocab
# preeprocessing, uncased, with BlueBERT vocabulary
python prepro_mediqa.py --biobert_vocab --cased

# preprocess the ground truth files for evaluation
python get_mediqa_gt_processed.py