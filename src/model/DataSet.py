from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declarative_base  # 모듈 변경

Base = declarative_base()


class DataSet(Base):
    __tablename__ = 'DataSet'

    dataSetId = Column(String(50), primary_key=True)
    dataDesc = Column(String(255))
    createUser = Column(String(50))
    isCreated = Column(DateTime)
    isUpdated = Column(DataTime)  # 기본값 설정
