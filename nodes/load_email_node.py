from utils.load_contact_frequencies import load_contact_frequencies

class LoadEmailNode:
    """
    이메일 데이터를 로드하고 연락처별 빈도수를 계산하는 노드
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.frequencies = {}

    def process(self, shared_store):
        """
        데이터베이스에서 이메일 데이터를 로드하고 연락처 빈도수를 계산합니다.
        
        Args:
            shared_store (dict): 데이터를 공유하기 위한 저장소
        
        Returns:
            dict: 업데이트된 shared_store
        """
        contact_frequencies = load_contact_frequencies(self.db_path)
        print(f"로드된 연락처 빈도수: {len(contact_frequencies)}")
        for contact, freq in contact_frequencies.items():
            print(f"연락처: {contact}, 빈도수: {freq}")
        shared_store['contact_frequencies'] = contact_frequencies
        return shared_store

if __name__ == "__main__":
    db_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\temp_Portable_search_DB_1062개.db"
    node = LoadEmailNode(db_path)
    shared_store = {}
    updated_store = node.process(shared_store)
    print("연락처별 빈도수:")
    for contact, freq in updated_store['contact_frequencies'].items():
        print(f"연락처: {contact}, 빈도수: {freq}")
