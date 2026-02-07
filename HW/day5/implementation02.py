import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "homework_01"

EMBEDDING_API_URL = "http://ws-04.wade0426.me/embed"

def get_embeddings_from_api(texts):
    """
    呼叫外部 API 將文字轉為向量
    """
    if isinstance(texts, str):
        texts = [texts]

    payload = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }

    try:
        response = requests.post(EMBEDDING_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result['embeddings']
    except Exception as e:
        print(f"❌ Embedding 轉換失敗: {e}")
        return []


documents = [
    "蘋果富含維他命C，對健康很好。",
    "特斯拉是一家知名的電動車公司。",
    "Python 是一種非常熱門的程式語言。",
    "今天天氣很好，適合去公園散步。",
    "人工智慧正在改變我們的工作方式。"
]

print("🔄 正在將文字轉換為向量...")
vectors = get_embeddings_from_api(documents)

if not vectors:
    print("程式終止：無法取得向量")
    exit()

# 取得向量維度 (通常是 384, 768, 1536 或 4096，依 API 模型而定)
# 這樣做的好處是不用猜測維度，直接看 API 回傳多長
vector_size = len(vectors[0])
print(f"✅ 取得向量成功，維度為: {vector_size}")

# ==========================================
# 步驟 1: 建立 Qdrant Collection 並連接
# (因為需要知道維度，所以通常會先試跑一次 Embedding 再建 Collection)
# ==========================================
print(f"🛠 正在建立 Collection: {COLLECTION_NAME}...")

# 使用 recreate_collection，如果已經存在會刪除重建 (方便反覆測試)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)


points_to_upsert = []

for i, (text, vector) in enumerate(zip(documents, vectors)):
    point = PointStruct(
        id=i + 1,  # ID 從 1 開始
        vector=vector,
        payload={"text": text, "category": "homework_data"}
    )
    points_to_upsert.append(point)

operation_info = client.upsert(
    collection_name=COLLECTION_NAME,
    points=points_to_upsert
)
print(f"💾 資料寫入狀態: {operation_info.status}")

# ==========================================
# 步驟 5: 召回內容 (Recall / Search)
# 測試：我們搜尋「電腦」相關的內容，看能不能找到 Python 或 AI
# ==========================================
query_text = "我想學習寫程式"
print(f"\n🔍 搜尋測試：'{query_text}'")

# 1. 將查詢語句也轉成向量
query_vector = get_embeddings_from_api([query_text])[0]

# 2. 進行相似度搜尋
search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3  # 只找最像的前 3 筆
)

# 3. 顯示結果
print("📊 搜尋結果：")
for result in search_results.points:
    print(f"Score: {result.score:.4f} | 內容: {result.payload['text']}")