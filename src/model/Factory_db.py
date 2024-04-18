from datetime import datetime  # datetime 모듈 임포트

from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declarative_base  # 모듈 변경

Base = declarative_base()


class Factory_db(Base):
    __tablename__ = 'Factory'

    dataSetId = Column(String(50), primary_key=True)
    factoryId = Column(String(50), primary_key=True)
    factoryDesc = Column(String(50))
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
    isCreated = Column(DateTime, server_default='CURRENT_TIMESTAMP')
