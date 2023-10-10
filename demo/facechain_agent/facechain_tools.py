from modelscope_agent.tools import Tool, TextToImageTool
from langchain.embeddings import ModelScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from langchain.text_loader import TextLoader


class StyleSearchTool(Tool):
    description = ''
    name = 'style_search'
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的想要的风格文本',
        'required': True
    }]

    def __init__(self, filepath: str, model_id: str ):
        self.embeddings = ModelScopeEmbeddings(model_id=model_id) 
        self.db = self.build_database(filepath)  # 使用FAISS构建文档嵌入向量数据库
        #self.folder_path=folder_path
        super().__init__()

    def build_database(self,filepath):
        docs=[]
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = CharacterTextSplitter()
        docs += (loader.load_and_split(textsplitter))
        db = FAISS.from_documents(docs, self.embeddings)
        return db

    def _remote_call(self, text):
        query_vector = self.embeddings.encode([text])[0]  # 将查询文本转换为嵌入向量
        similar_doc_indices, _ = self.db.search([query_vector], k=1)  # 查找最相似的文档

        # # 获取最相似的文档的name字段
        # best_match_name = self.get_name_from_document_index(similar_doc_indices[0][0])
        best_match_name = similar_doc_indices[0][0]
        result = {
            'name': self.name,
            'best_match_name': best_match_name
        }
        return {'result': result}

    def _local_call(self, text):
        query_vector = self.embeddings.encode([text])[0]  # 将查询文本转换为嵌入向量
        similar_doc_indices, _ = self.db.search([query_vector], k=1)  # 查找最相似的文档

        # 获取最相似的文档的name字段
        #best_match_name = self.get_name_from_document_index(similar_doc_indices[0][0])
        best_match_name = similar_doc_indices[0][0]
        result = {
            'name': self.name,
            'best_match_name': best_match_name
        }
        return {'result': result}