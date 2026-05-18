"""Тесты csv_writer."""
from mgeval.csv_writer import write_mgeval_csv


def test_csv_writer_basic_format(tmp_path):
    rows = [
        {"feature": "total_used_pitch", "model": "cmt", "kl": 1.234567,
         "oa": 0.5, "n_real_pieces": 100, "n_gen_pieces": 200},
        {"feature": "avg_ioi", "model": "mingus", "kl": 0.123456,
         "oa": 0.9876, "n_real_pieces": 100, "n_gen_pieces": 200},
    ]
    out = tmp_path / "mgeval.csv"
    write_mgeval_csv(rows, out)
    assert out.exists()
    content = out.read_text()
    assert "feature,model,kl,oa,n_real_pieces,n_gen_pieces" in content
    assert "total_used_pitch,cmt,1.234567,0.500000,100,200" in content
    assert "avg_ioi,mingus,0.123456,0.987600,100,200" in content


def test_csv_writer_atomic_no_tmp_after_success(tmp_path):
    rows = [
        {"feature": "f1", "model": "cmt", "kl": 1.0,
         "oa": 0.5, "n_real_pieces": 10, "n_gen_pieces": 20},
    ]
    out = tmp_path / "mgeval.csv"
    write_mgeval_csv(rows, out)
    assert out.exists()
    assert not (tmp_path / "mgeval.csv.tmp").exists()


def test_csv_writer_creates_parent_dir(tmp_path):
    out = tmp_path / "subdir" / "deeper" / "mgeval.csv"
    rows = [
        {"feature": "f1", "model": "cmt", "kl": 1.0,
         "oa": 0.5, "n_real_pieces": 10, "n_gen_pieces": 20},
    ]
    write_mgeval_csv(rows, out)
    assert out.exists()
