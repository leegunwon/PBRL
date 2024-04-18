from datetime import datetime  # datetime 모듈 임포트

from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.orm import declarative_base  # 모듈 변경

Base = declarative_base()


class Event_db(Base):
    __tablename__ = 'Event'
    simulationNumber = Column(String(50), primary_key=True)
    dataSetId = Column(String(50), primary_key=True)
    eventId = Column(String(50), primary_key=True)
    jobId = Column(String(50))
    jobType = Column(String(50))
    jobOper = Column(String(50))
    machine = Column(String(50))
    startTime = Column(Integer)
    endTime = Column(Integer)
    eventType = Column(String(50))
    ruleName = Column(String(50))
    stepNum = Column(Integer)
    setupTime = Column(Integer)
    excessQTime = Column(Integer)
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
    isCreated = Column(DateTime, server_default='CURRENT_TIMESTAMP')

    def __init__(self, event, dataset_id, simulation_time):
        event: Event
        self.simulationNumber = simulation_time
        self.dataSetId = dataset_id
        self.eventId = event.id
        self.jobId = event.job.id if event.job != None else None
        self.jobType = event.job.job_type if event.job != None else None
        self.jobOper = event.jop
        print(event.event_type)
        self.machine = None if event.event_type == "job_arrival" or event.event_type == "plan_end" else event.machine.id

        self.startTime = event.start_time
        self.endTime = event.end_time
        self.eventType = event.event_type
        self.ruleName = event.rule_name
        self.stepNum = event.step_num if event.step_num != "NONE" else None
        self.setupTime = event.setup_table if event.setup_table != "NONE" else None
        self.excessQTime = event.q_time_diff if event.q_time_diff != "None" else None
