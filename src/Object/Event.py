from master_db.DB_query import *

from src.model.event_db import *


class Event(object):
    id_counter = 0

    @classmethod
    def generate_id(cls):
        cls.id_counter += 1
        return cls.id_counter

    # Default Constructor
    def __init__(self, job, jop, machine, start_time, end_time, event_type, rule_name, step_num, setup_time,
                 q_time_diff):
        # 고정 정보
        self.id = Event.generate_id()  # 자동 생성되는 event_id
        self.job = job
        self.jop = jop
        self.machine = machine
        self.start_time = start_time
        self.end_time = end_time
        self.event_type = event_type
        self.rule_name = rule_name
        self.step_num = step_num
        self.setup_table = setup_time
        self.q_time_diff = q_time_diff

        if self.q_time_diff == 0 or self.q_time_diff == "None":
            self.q_time_check = True
            # print("D")
        else:
            self.q_time_check = False

    def send_db(self, datasetId, simulation_time):
        data_to_insert = Event_db(self, datasetId, simulation_time)
        DB_query.set_event_history(data_to_insert)
