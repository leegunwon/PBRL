import copy
import time

from src.simulator.Simulator import *


class GA_FJSP:
    population = []  # [ [job_seq,mac_seq, fittness], ...]
    MUTATION_PROB = 0.05
    POP_SIZE = 200
    RANGE = 0
    NUM_OFFSPRING = 5
    SELECTION_PRESSURE = 2
    END = 0.9
    job_type_list = []
    generation = 0  # 현재 세대 수

    @classmethod
    def init_job_seq_random(cls):
        # 시뮬레이터로 부터 job정보 가져오기
        job_seq = Simulator.get_job_seq()
        random.shuffle(job_seq)
        return job_seq

    @classmethod
    def init_mac_seq_random(cls, job_seq):
        machine_lists = []
        for job in job_seq:
            oper = Simulator.lot_list[job].current_operation_id
            Simulator.lot_list[job].oper_check_for_meta()
            machine_list = []
            for machine in Simulator.Processing_time_table[oper]:
                p_time = Simulator.Processing_time_table[oper][machine]
                if p_time != 0:
                    machine_list.append(machine)
            machine_lists.append(machine_list)

        machine_seq = [random.choice(machine_list) for machine_list in machine_lists if machine_list]
        return machine_seq

    @classmethod
    def init_population(cls):
        for i in range(cls.POP_SIZE):
            job_seq = cls.init_job_seq_random()
            mac_seq = cls.init_mac_seq_random(job_seq)
            cls.job_type_list = list(set(job_seq))
            fittness = Simulator.get_fittness_with_meta_heuristic(job_seq, mac_seq)
            cls.population.append([job_seq, mac_seq, fittness])

    @classmethod
    def crossover_operator_MOX(cls, mom, dad):
        mom_ch = copy.deepcopy(mom[0])
        dad_ch = copy.deepcopy(dad[0])

        from_mom_to_off_index = []
        from_dad_to_off_index = []

        mom_ch_jobs = []
        dad_ch_jobs = []

        mom_ch_cp = copy.deepcopy(mom_ch)
        dad_ch_cp = copy.deepcopy(dad_ch)

        # point 2개 찍기
        point1 = random.randint(0, len(mom_ch) - 1)
        point2 = random.randint(point1 + 1, len(mom_ch))

        for point in range(point1, point2):
            for j in range(len(dad_ch)):
                if mom_ch[point] == dad_ch_cp[j]:  # 엄마의 포인트 1 지점과, 아빠의 j번째가 같은 경우
                    dad_ch_cp[j] = -1
                    from_mom_to_off_index.append(j)  # 그 인덱스와, 어떤 job이었는지 기록
                    mom_ch_jobs.append(mom_ch[point])
                    break  # 기록 완료시 break함
            for k in range(len(dad_ch)):
                if dad_ch[point] == mom_ch_cp[k]:  # 아빠의 포인트 지점과 엄마의 k가 같은지 확인
                    mom_ch_cp[k] = -1
                    from_dad_to_off_index.append(k)  # 그 인덱스와 어떤 job이었는지 기록
                    dad_ch_jobs.append(dad_ch[point])
                    break

        offspring_from_dad = copy.deepcopy(dad_ch)  # 자녀해 생성 1
        offspring_from_mom = copy.deepcopy(mom_ch)  # 자녀해 생성 2

        from_mom_to_off_index.sort(reverse=False)  # index 섞기, 앞 순서부터 채워져 있음, ex) [1,5,11]
        from_dad_to_off_index.sort(reverse=False)

        for i in range(len(from_dad_to_off_index)):
            offspring_from_dad[from_mom_to_off_index[i]] = mom_ch_jobs[i]  # 아빠거 배낀거 1번에 엄마꺼에서 뽑은 job넣어줌
            offspring_from_mom[from_dad_to_off_index[i]] = dad_ch_jobs[i]

        offspring_from_dad_ma = []
        offspring_from_mom_ma = []

        for i in range(len(offspring_from_mom)):
            for j in range(len(offspring_from_mom)):
                if offspring_from_dad[i] == dad_ch[j]:  # 새로 생긴 자식과 아빠가 비교함
                    offspring_from_dad_ma.append(dad[1][j])
                    dad_ch[j] = -1

                    break
            for k in range(len(offspring_from_mom)):
                if offspring_from_mom[i] == mom_ch[k]:
                    offspring_from_mom_ma.append(mom[1][k])
                    mom_ch[k] = -1
                    break
        fitness_mom = Simulator.get_fittness_with_meta_heuristic(offspring_from_mom, offspring_from_mom_ma)
        fitness_dad = Simulator.get_fittness_with_meta_heuristic(offspring_from_dad, offspring_from_dad_ma)

        offspring1 = [offspring_from_mom, offspring_from_mom_ma, fitness_mom]
        offspring2 = [offspring_from_dad, offspring_from_dad_ma, fitness_dad]
        return offspring1, offspring2

    @classmethod
    def crossover_operator_JCO(cls, mom, dad):
        mom_ch = copy.deepcopy(mom[0])
        dad_ch = copy.deepcopy(dad[0])
        Simulator.get_job_seq()

    @classmethod
    def mutation_operator(cls, offspring):
        for i in range(len(offspring[0])):
            coin = random.random()
            if coin < 0.01:
                offspring[1][i] = Simulator.get_random_machine(offspring[0][i])
            # else:
            # offspring[1][i] = Simulator.get_shortest_processing_time_machine()
            Simulator.lot_list[offspring[0][i]].oper_check_for_meta()
        return offspring

    @classmethod
    def sort_population(cls):
        cls.population.sort(key=lambda x: x[2], reverse=False)

    @classmethod
    def selection_operator(cls):
        # 룰렛 휠
        inverse_fitness = [1 / x[2] for x in cls.population]
        chrom = random.choices(cls.population, weights=inverse_fitness, k=2)
        mom_ch = chrom[0]
        dad_ch = chrom[1]
        return mom_ch, dad_ch

    @classmethod
    def replacement_operator(cls, offsprings):
        result_population = []
        for i in range(cls.NUM_OFFSPRING):
            cls.population.pop()
        for i in range(cls.NUM_OFFSPRING):
            cls.population.append(offsprings[i])

    @classmethod
    def print_avg_fittness(cls):
        sum_fitness = sum(i[2] for i in cls.population)
        avg_fitness = sum_fitness / len(cls.population)
        print(f"세대 수 : {cls.generation} 최고 fitness : {cls.population[0][2]} 평균 fitness : {avg_fitness}")

    @classmethod
    def search(cls):
        population = []  # 해집단
        offsprings = []  # 자식해집단

        cls.init_population()
        cls.sort_population()
        while True:
            offsprings = []
            count_end = 0  # 동일 갯수
            for i in range(cls.NUM_OFFSPRING):

                mom_ch, dad_ch = cls.selection_operator()
                st = time.time()
                offspring1, offspring2 = cls.crossover_operator_MOX(mom_ch, dad_ch)
                et = time.time()
                # print(f"cross_over_time = {et- st}")
                coin = random.random()
                if coin < cls.MUTATION_PROB:
                    offspring1 = cls.mutation_operator(offspring1)
                    offspring2 = cls.mutation_operator(offspring2)
                offsprings.append(offspring1)
                offsprings.append(offspring2)

            cls.replacement_operator(offsprings)
            cls.sort_population()
            cls.print_avg_fittness()
            cls.generation += 1

            if cls.generation == 10000:
                break

        print(cls.population[0])
