class ReportNode:
    """
    이메일 요약을 기반으로 보고서를 생성하는 노드
    """
    def __init__(self, output_path):
        self.output_path = output_path
        self.report = ""

    def process(self, shared_store):
        """
        이메일 요약을 기반으로 보고서를 생성하고 파일로 저장합니다.
        
        Args:
            shared_store (dict): 데이터를 공유하기 위한 저장소
        
        Returns:
            dict: 업데이트된 shared_store
        """
        email_summaries_data = shared_store.get('email_summaries', {})
        contact_frequencies = shared_store.get('contact_frequencies', {})
        print(f"보고서 생성 시작 - 요약된 연락처 수: {len(email_summaries_data)}")
        self.report = "# 이메일 분석 보고서\n\n"
        
        if not email_summaries_data:
            self.report += "처리할 이메일 요약 정보가 없습니다.\n"
        else:
            for contact, summary_data in email_summaries_data.items():
                freq = contact_frequencies.get(contact, 0)
                self.report += f"## 연락처: {contact}\n"
                self.report += f"- **연락 빈도수**: {freq}회\n"
                
                individual_csv_path = summary_data.get('individual_csv_path', 'N/A')
                overall_summary_text = summary_data.get('overall', '종합 요약 정보 없음.')

                self.report += f"- **개별 이메일 요약 CSV**: `{individual_csv_path}`\n"
                self.report += f"### 종합 요약:\n"
                self.report += f"{overall_summary_text}\n\n"
        
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(self.report)
            print(f"보고서가 성공적으로 저장되었습니다: {self.output_path}")
        except Exception as e:
            print(f"보고서 저장 중 오류 발생: {e}")
            # Optionally, re-raise or handle more gracefully
            # raise
        
        shared_store['report_content'] = self.report # Changed key to avoid confusion with a 'report' object
        shared_store['report_path'] = self.output_path
        return shared_store

if __name__ == "__main__":
    output_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\email_report.md"
    node = ReportNode(output_path)
    shared_store = {'email_summaries': {'example@domain.com': {'overall': 'This is a summary of emails with example@domain.com', 'individual_csv_path': 'path/to/example.csv'}}, 'contact_frequencies': {'example@domain.com': 5}}
    updated_store = node.process(shared_store)
    print("보고서 생성 완료:")
    print(updated_store['report_content'])
