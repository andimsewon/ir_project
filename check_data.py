"""데이터 파일 확인 스크립트"""
import os

data_dir = "data"

print("=" * 50)
print("데이터 파일 확인")
print("=" * 50)

doc_path = os.path.join(data_dir, "documents.tsv")
if os.path.exists(doc_path):
    size_mb = os.path.getsize(doc_path) / (1024 * 1024)
    print(f"\ndocuments.tsv:")
    print(f"  존재: 예")
    print(f"  크기: {size_mb:.2f} MB")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f):
            if i < 3:
                lines.append(line.strip())
            if i >= 100:  # 처음 100줄만 읽어서 라인 수 추정
                break
        print(f"  샘플 라인:")
        for line in lines:
            print(f"    {line[:80]}...")
else:
    print(f"\ndocuments.tsv: 존재하지 않음")

for split in ["training", "validation", "test"]:
    qpath = os.path.join(data_dir, f"queries_{split}.tsv")
    if os.path.exists(qpath):
        size_kb = os.path.getsize(qpath) / 1024
        with open(qpath, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f) - 1  # 헤더 제외
        print(f"\nqueries_{split}.tsv:")
        print(f"  존재: 예")
        print(f"  크기: {size_kb:.2f} KB")
        print(f"  쿼리 수: {count}")
    else:
        print(f"\nqueries_{split}.tsv: 존재하지 않음")

for split in ["training", "validation", "test"]:
    qpath = os.path.join(data_dir, f"qrels_{split}.tsv")
    if os.path.exists(qpath):
        size_kb = os.path.getsize(qpath) / 1024
        with open(qpath, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f) - 1  # 헤더 제외
        print(f"\nqrels_{split}.tsv:")
        print(f"  존재: 예")
        print(f"  크기: {size_kb:.2f} KB")
        print(f"  qrels 수: {count}")
    else:
        print(f"\nqrels_{split}.tsv: 존재하지 않음")

print("\n" + "=" * 50)






