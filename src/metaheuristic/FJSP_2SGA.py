from src.metaheuristic.FJSP_GA import *


class FJSP_2SGA(GA_FJSP):

    @classmethod
    def init_MA_LS(cls, scheduling_seq):
        routing_seq = []
        for job in scheduling_seq:
            machine = Simulator.get_least_time_machine(job)
            routing_seq.append(machine)
        Simulator.reset()
        return routing_seq

    @classmethod
    def jco(cls, dad, mom):
        dad_ch = copy.deepcopy(dad[0])
        mom_ch = copy.deepcopy(mom[0])
        job_seq = Simulator.job_info.keys()
        change_job_list = []
        for job in job_seq:
            coin = random.random()
            if coin < 0.5:
                change_job_list.append(job)

        # j01, j02, j01,j02,j03,j04,j01,j02
        # j01 ,j03
        # j04, j01, j02, j03, j01, j02, j01, j02
        for i in range(len(dad_ch)):
            if dad_ch[i] in change_job_list:
                dad_ch[i] = None
            else:
                mom_ch.remove(dad_ch[i])
        for i in range(len(dad_ch)):
            if dad_ch[i] == None:
                dad_ch[i] = mom_ch.pop(0)

        ma_offspring = cls.init_MA_LS(dad_ch)
        makespan = Simulator.get_fittness_with_meta_heuristic(dad_ch, ma_offspring)
        offspring = [dad_ch, ma_offspring, makespan]
        return offspring

    @classmethod
    def sco(cls, dad, mom):
        dad_ch = copy.deepcopy(dad[0])
        mom_ch = copy.deepcopy(mom[0])
        point = random.randint(0, len(dad_ch))
        for i in range(point):
            mom_ch.remove(dad_ch[i])
        for i in range(point, len(dad_ch)):
            dad_ch[i] = mom_ch.pop(0)

        ma_offspring = cls.init_MA_LS(dad_ch)
        makespan = Simulator.get_fittness_with_meta_heuristic(dad_ch, ma_offspring)
        offspring = [dad_ch, ma_offspring, makespan]
        return offspring

    @classmethod
    def OSM(cls, offspring):  # two point reverse method
        new_os = copy.deepcopy(offspring[0])
        numbers = list(range(len(offspring[0])))
        first_element = random.choice(numbers)
        numbers.remove(first_element)
        second_element = random.choice(numbers)

        new_os[first_element], new_os[second_element] = new_os[second_element], new_os[first_element]

        ma_offspring = cls.init_MA_LS(new_os)
        makespan = Simulator.get_fittness_with_meta_heuristic(new_os, ma_offspring)
        offspring = [new_os, ma_offspring, makespan]

        return offspring

    @classmethod
    def aco(cls, dad, mom):
        dad_ch = copy.deepcopy(dad[0])
        mom_ch = copy.deepcopy(mom[0])
        new_ma = []

        for i in range(len(dad_ch)):
            for j in range(len(mom_ch)):
                if dad_ch[i] == mom_ch[j]:
                    mom_ch[j] = -1
                    new_ma.append(mom[1][j])
                    break

        makespan = Simulator.get_fittness_with_meta_heuristic(dad_ch, new_ma)
        offspring = [dad_ch, new_ma, makespan]
        return offspring

    @classmethod
    def init_population2(cls):
        for i in range(cls.POP_SIZE):
            job_seq = cls.init_job_seq_random()
            mac_seq = cls.init_MA_LS(job_seq)
            cls.job_type_list = list(set(job_seq))
            fittness = Simulator.get_fittness_with_meta_heuristic(job_seq, mac_seq)
            cls.population.append([job_seq, mac_seq, fittness])

    @classmethod
    def search(cls):
        population = []  # 해집단
        offsprings = []  # 자식해집단

        cls.init_population2()
        cls.sort_population()
        while True:
            offsprings = []
            count_end = 0  # 동일 갯수
            for i in range(cls.NUM_OFFSPRING):
                mom_ch, dad_ch = cls.selection_operator()
                coin = random.random()
                if coin < 0.7:
                    offspring = cls.jco(mom_ch, dad_ch)
                else:
                    offspring = cls.aco(mom_ch, dad_ch)
                    # offspring = cls.sco(mom_ch, dad_ch)

                coin2 = random.random()
                if coin2 < 0.05:
                    offspring = cls.OSM(offspring)
                    print("변이")
                offsprings.append(offspring)

            cls.replacement_operator(offsprings)
            cls.sort_population()
            cls.print_avg_fittness()
            cls.generation += 1

            if cls.generation == 10000:
                break

        print(cls.population[0])
