import numpy as np
import pandas as pd
import re
import torch
from src.Object.Lot import *
from src.learner.common.Hyperparameters import *

class StateManager:
    state_time = 0

    @classmethod
    def get_state(cls, j_list, r_list, cur_runtime, number_of_job, bucket, oper_in_list):
        if Hyperparameters.state_type == "state_12":
            s = cls.set_state_12(j_list, r_list, cur_runtime)
        elif Hyperparameters.state_type == "state_36":
            s = cls.set_state_36(j_list, r_list, cur_runtime, number_of_job)
        elif Hyperparameters.state_type == "action_masking_state":
            s = cls.set_action_masking_state(j_list, r_list, cur_runtime, number_of_job)
        elif Hyperparameters.state_type == "cnn_state":
            s = cls.set_state_cnn(bucket, oper_in_list, cur_runtime)
        elif Hyperparameters.state_type == "cnn":
            s = cls.set_state_cnn_state(bucket, oper_in_list, cur_runtime)
        elif Hyperparameters.state_type == 'default_state':
            s = cls.set_state_default(j_list, r_list, cur_runtime)
        elif Hyperparameters.state_type == 'pbrl_state':
            s = cls.set_pbrl_simple(j_list, r_list)
        return s

    @classmethod
    def set_state_cnn_state(cls, bucket, oper_in_list, cur_time):
        # cls.state_time = curr_time
        oper_in_list['cur'] = 0

        curr_bucket = cur_time // (7 * 24) if cur_time != 0 else 0
        cur_bucket_list = [0 for i in range(5)]
        curr_bucket = min(4, curr_bucket)
        cur_bucket_list[int(curr_bucket)] = 20

        df_bucket = pd.DataFrame(bucket)
        df_bucket['cur'] = cur_bucket_list
        df_bucket = df_bucket.T

        s_oper_in_list = pd.Series(oper_in_list)
        concat = pd.concat([df_bucket, s_oper_in_list], axis=1)
        normal_concat = concat.div(20)
        s = normal_concat.to_numpy()
        tensor_data = s[np.newaxis, :, :]
        # print(tensor_data)
        return tensor_data

    @classmethod
    def set_state_cnn(cls, bucket, oper_in_list, cur_time):
        # cls.state_time = curr_time
        s = []
        for job, time_bucket_dict in bucket.items():
            for demand_qty in time_bucket_dict.values():
                s.append(demand_qty / 20)
        for job_id in oper_in_list:
            qty = oper_in_list[job_id]
            s.append(qty / 20)
        runtime = [0 for i in range(5)]
        curr_bucket = cur_time // (7 * 24) if cur_time != 0 else 0
        curr_bucket = min(4, curr_bucket)
        runtime[int(curr_bucket)] = 1

        s += runtime

        # print(s)
        df = pd.Series(s)
        s = df.to_numpy()

        return s

    @classmethod
    def set_state_default(cls, j_list, r_list, curr_time):
        # 작업 대기 비율 (number_of_jobs_wait/len(j_list)): 전체 작업 중에서 대기 상태인 작업의 비율입니다.
        # 작업 처리 중 비율 (number_of_jobs_load/len(j_list)): 전체 작업 중에서 처리 중인 (PROCESSING 상태인) 작업의 비율입니다.
        # 대기 중인 작업의 평균 지연 시간 (total_tardiness / number_of_jobs_wait): 대기 중인 작업의 지연 시간의 평균입니다. 단, 대기 중인 작업이 없을 경우 0을 추가합니다.
        # 대기 중인 작업의 평균 흐름 시간 (total_flow_time / number_of_jobs_wait): 대기 중인 작업의 흐름 시간의 평균입니다. 흐름 시간은 작업이 시작된 후 현재 시간까지의 경과 시간입니다. 대기 중인 작업이 없으면 0을 추가합니다.
        # 현재 시간 대비 최대 예약 시간의 비율 (current_time / max_reservation_time): 기계에 예약된 작업 중 가장 늦은 예약 시간 대비 현재 시간의 비율입니다. 만약 예약된 작업이 없다면 0을 추가합니다.
        # 기계별 예약 시간 차이의 평균 (total_reservation_time_diff / len(r_list)): 모든 기계에 대한 예약 시간과 현재 시간의 차이의 평균입니다.
        # 작업 완료 비율 (number_of_job_done/len(j_list)): 전체 작업 중에서 완료된 작업의 비율입니다.
        # 완료된 작업의 평균 지연 시간 (total_job_tardiness_done / number_of_job_done): 완료된 작업의 지연 시간의 평균입니다. 완료된 작업이 없을 경우 0을 추가합니다.

        s = []
        number_of_jobs_wait = 0  # clear
        number_of_jobs_load = 0  # clear
        total_tardiness = 0
        total_flow_time = 0
        number_of_job_done = 0  # clear

        total_job_tardiness_done = 0  # clear
        total_job_q_time_over_done = 0  # clear
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                number_of_jobs_wait += 1
                total_tardiness += j_list[job].cal_tardiness(curr_time)
                total_flow_time += j_list[job].cal_flowtime(curr_time)
            elif j_list[job].status == "PROCESSING":
                number_of_jobs_load += 1
            elif j_list[job].status == "DONE":
                number_of_job_done += 1
                total_job_tardiness_done += j_list[job].tardiness_time

        current_time = curr_time
        total_reservation_time_diff = 0
        max_reservation_time = 0
        for machine in r_list:
            total_reservation_time_diff += r_list[machine].reservation_time - current_time
            if max_reservation_time > r_list[machine].reservation_time:
                max_reservation_time = r_list[machine].reservation_time

        s.append(number_of_jobs_wait / len(j_list))
        s.append(number_of_jobs_load / len(j_list))
        if number_of_jobs_wait == 0:
            for _ in range(2):
                s.append(0)
        else:
            s.append(total_tardiness / number_of_jobs_wait)
            s.append(total_flow_time / number_of_jobs_wait)

        if max_reservation_time == 0:
            s.append(0)
        else:
            s.append(current_time / max_reservation_time)
        s.append(total_reservation_time_diff / len(r_list))

        s.append(number_of_job_done / len(j_list))
        if number_of_job_done == 0:
            s.append(0)
        else:
            s.append(total_job_tardiness_done / number_of_job_done)

        df = pd.Series(s)
        s = df.to_numpy()

        return s

    @classmethod
    def set_state_12(cls, j_list, r_list, cuur_time):
        cls.state_time = cuur_time
        """
                재공 정보 :
                    대기 중인 job들의 개수
                    작업 중인 job들의 개수
                    대기 중인 job들의 남은 operation 개수 평균
                    대기 중인 job들의 tardiness 평균
                    대기 중인 job들의 q-time 초과 평균
                    대기 중인 job들의 flow time 평균

                기계 정보 :
                    기계의 현재 시간
                    현재 시간 / 다른 기계의 최대 시간
                    다른 기계들과 차이의 평균

                누적 정보 :
                    현재까지 total tardiness
                    현재까지 total q over time
                    현재까지 처리한 job 개수
                """
        s = []
        number_of_jobs_wait = 0  # clear
        number_of_jobs_load = 0  # clear
        total_remain_operation = 0
        total_tardiness = 0
        total_q_time_over = 0
        total_flow_time = 0
        number_of_job_done = 0  # clear

        total_job_tardiness_done = 0  # clear
        total_job_q_time_over_done = 0  # clear
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                number_of_jobs_wait += 1
                total_remain_operation += j_list[job].remain_operation
                total_tardiness += j_list[job].cal_tardiness(cuur_time)
                total_q_time_over += j_list[job].cal_q_time(cuur_time)
                total_flow_time += j_list[job].cal_flowtime(cuur_time)
            elif j_list[job].status == "PROCESSING":
                number_of_jobs_load += 1
            elif j_list[job].status == "DONE":
                number_of_job_done += 1
                total_job_tardiness_done += j_list[job].tardiness_time
                q_total = j_list[job].cal_q_time_total()
                total_job_q_time_over_done += q_total

        current_time = cuur_time
        total_reservation_time_diff = 0
        max_reservation_time = 0
        for machine in r_list:
            total_reservation_time_diff += r_list[machine].reservation_time - current_time
            if max_reservation_time > r_list[machine].reservation_time:
                max_reservation_time = r_list[machine].reservation_time

        s.append(number_of_jobs_wait)
        s.append(number_of_jobs_load)
        if number_of_jobs_wait == 0:
            for _ in range(4):
                s.append(0)
        else:
            s.append(total_remain_operation / number_of_jobs_wait)
            s.append(total_tardiness / number_of_jobs_wait)
            s.append(total_q_time_over / number_of_jobs_wait)
            s.append(total_flow_time / number_of_jobs_wait)

        s.append(current_time)
        if max_reservation_time == 0:
            s.append(0)
        else:
            s.append(current_time / max_reservation_time)
        s.append(total_reservation_time_diff / len(r_list))

        s.append(number_of_job_done)
        if number_of_job_done == 0:
            s.append(0)
            s.append(0)
        else:
            s.append(total_job_tardiness_done / number_of_job_done)
            s.append(total_job_q_time_over_done / number_of_job_done)

        df = pd.Series(s)
        s = df.to_numpy()

        return s

    @classmethod
    def set_state_36(cls, j_list, r_list, cur_time, number_of_job_type):
        cls.state_time = cur_time
        """
        len(1) :
        stocker job 개수 / 전체 job
        processing job 개수 / 전체 job
        종료된 job 개수 / 전체 job
        setting setup type length / job type length
        q time over job in stocker / stocker jobs
        q time safe job in stocker / stocker jobs
        
        len(# of machines) :
        각 기계의 util
        각 기계의 r_time / max_r_time
        각 기계의 l_time / max_l_time    
        """
        s = []
        number_of_jobs = len(j_list)
        number_of_machines = len(r_list)
        number_of_job_wait = 0
        number_of_job_load = 0
        number_of_job_done = 0
        q_time_over_job = 0
        q_time_safe_job = 0

        number_of_jobs_t = {}
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                number_of_job_wait += 1
                check_q_time = j_list[job].check_q_time(cur_time)
                if check_q_time == 0:
                    q_time_over_job += 1
                elif check_q_time == 1:
                    q_time_safe_job += 1
                else:
                    pass
            elif j_list[job].status == "PROCESSING":
                number_of_job_load += 1
            elif j_list[job].status == "DONE":
                number_of_job_done += 1
        s.append(number_of_job_wait / number_of_jobs)  # 1
        s.append(number_of_job_load / number_of_jobs)  # 2
        s.append(number_of_job_done / number_of_jobs)  # 3

        s.append((q_time_over_job / number_of_job_wait) if number_of_job_wait > 0 else 0)  # 4
        s.append((q_time_safe_job / number_of_job_wait) if number_of_job_wait > 0 else 0)  # 5

        max_last_work_finish_time = 0
        max_reservation_time = 0
        reservation_list = []
        lwft_list = []  # last work finish time
        setup_status_list = []
        for machine in r_list:
            util = r_list[machine].cal_util2()
            s.append(util)  # 7~16
            reservation_list.append(r_list[machine].reservation_time)
            lwft_list.append(r_list[machine].last_work_finish_time)
            setup_status_list.append(r_list[machine].setup_status)
            if max_reservation_time > r_list[machine].reservation_time:
                max_reservation_time = r_list[machine].reservation_time
            if max_last_work_finish_time > r_list[machine].last_work_finish_time:
                max_last_work_finish_time = r_list[machine].last_work_finish_time

        s.append(len(set(setup_status_list)) / number_of_job_type)  # 6

        for i in range(len(reservation_list)):
            s.append((reservation_list[i] / max_reservation_time) if max_reservation_time > 0 else 0)  # 17~26
            s.append((lwft_list[i] / max_last_work_finish_time) if max_last_work_finish_time > 0 else 0)  # 27~36

        df = pd.Series(s)
        s = df.to_numpy()

        return s

    @classmethod
    def set_action_masking_state(cls, j_list, r_list, cur_time, number_of_job_type):
        cls.state_time = cur_time
    @classmethod
    def change_job_type_to_num(cls, job_type):
        if re.search(r'j01', job_type):
            job_type_num = [0, 0, 1]
        elif re.search(r'j02', job_type):
            job_type_num = [0, 1, 0]
        elif re.search(r'j03', job_type):
            job_type_num = [0, 1, 1]
        elif re.search(r'j04', job_type):
            job_type_num = [1, 0, 0]
        elif re.search(r'j05', job_type):
            job_type_num = [1, 0, 1]
        else:
            job_type_num = [0, 0, 0]
        return job_type_num

    @classmethod
    def count_job_type(cls, job, counter):
        if job.job_type == 'j01':
            counter[0] += 1
        elif job.job_type == 'j02':
            counter[1] += 1
        elif job.job_type == 'j03':
            counter[2] += 1
        elif job.job_type == 'j04':
            counter[3] += 1
        elif job.job_type == 'j05':
            counter[4] += 1

        return counter

    @classmethod
    def set_pbrl_simple(cls, j_list, r_list):
        # 각 머신에 할당되어 있는 lot의 종료 시간
        # 각 머신에 할당된 job type
        # 각 job type 별 남은 개수
        # 각 job type 별 가장 due date가 빠른 job의 남은 시간
        s = []
        due_date_for_job_type = {'j01': 1000000000, 'j02': 1000000000, 'j03': 1000000000, 'j04': 1000000000, 'j05': 1000000000}
        counter = [0, 0, 0, 0, 0]
        for machine in r_list:
            s.append(r_list[machine].reservation_time)
        s = [round(x / 24, 4) for x in s]
        for machine in r_list:
            s += cls.change_job_type_to_num(r_list[machine].setup_status)

        for job in j_list.items():  # job 이름과 operation이름 찾기
            if job[1].status == "WAIT":
                counter = cls.count_job_type(job[1], counter)
                # 가장 납기가 얼마 남지 않은 job
                if due_date_for_job_type[job[1].job_type] > job[1].duedate:
                    due_date_for_job_type[job[1].job_type] = job[1].duedate

        due_date = [round(i/24, 4) if i != 1000000000 else 0 for i in due_date_for_job_type.values()]
        s += due_date
        s += counter
        # s.append(round(curr_time/24, 2))
        s = np.array(s)
        return s
