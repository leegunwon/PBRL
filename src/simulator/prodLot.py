class ProdLot:
    lot_list_by_prod = {}

    @classmethod
    def init_prod_lot(cls, lot_list):
        for lot_id in lot_list:
            if lot_id not in cls.lot_list_by_prod:
                cls.lot_list_by_prod[lot_list[lot_id].job_id] = []
            cls.lot_list_by_prod[lot_list[lot_id].job_id].append(lot_list[lot_id])
