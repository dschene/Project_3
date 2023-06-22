class Aql:

    def __init__(self):
        
        self.aql_dict = {2: {.065: None, .1: None, .15: None, .25: None, .4: None, .65: None, 1.0: None, 
                             1.5: None, 2.5: None, 4.0: None, 6.5: 0, 10.0: None, 15.0: None}, 
                         3: {.065: None, .1: None, .15: None, .25: None, .4: None, .65: None, 1.0: None, 
                             1.5: None, 2.5: None, 4.0: 0, 6.5: None, 10.0: None, 15.0: 1}, 
                         5: {.065: None, .1: None, .15: None, .25: None, .4: None, .65: None, 1.0: None, 
                             1.5: None, 2.5: 0, 4.0: None, 6.5: None, 10.0: 1, 15.0: 2}, 
                         8: {.065: None, .1: None, .15: None, .25: None, .4: None, .65: None, 1.0: None, 
                             1.5: 0, 2.5: None, 4.0: None, 6.5: 1, 10.0: 2, 15.0: 3}, 
                         13: {.065: None, .1: None, .15: None, .25: None, .4: None, .65: None, 1.0: 0, 
                             1.5: None, 2.5: None, 4.0: 1, 6.5: 2, 10.0: 3, 15.0: 5}, 
                         20: {.065: None, .1: None, .15: None, .25: None, .4: None, .65: 0, 1.0: None, 
                             1.5: None, 2.5: 1, 4.0: 2, 6.5: 3, 10.0: 5, 15.0: 7}, 
                         32: {.065: None, .1: None, .15: None, .25: None, .4: 0, .65: None, 1.0: None, 
                             1.5: 1, 2.5: 2, 4.0: 3, 6.5: 5, 10.0: 7, 15.0: 10}, 
                         50: {.065: None, .1: None, .15: None, .25: 0, .4: None, .65: None, 1.0: 1, 
                             1.5: 2, 2.5: 3, 4.0: 5, 6.5: 7, 10.0: 10, 15.0: 14}, 
                         80: {.065: None, .1: None, .15: 0, .25: None, .4: None, .65: 1, 1.0: 2, 
                             1.5: 3, 2.5: 5, 4.0: 7, 6.5: 10, 10.0: 14, 15.0: 21}, 
                         125: {.065: None, .1: 0, .15: None, .25: None, .4: 1, .65: 2, 1.0: 3, 
                             1.5: 5, 2.5: 7, 4.0: 10, 6.5: 14, 10.0: 21, 15.0: None}, 
                         200: {.065: 0, .1: None, .15: None, .25: 1, .4: 2, .65: 3, 1.0: 5, 
                             1.5: 7, 2.5: 10, 4.0: 14, 6.5: 21, 10.0: None, 15.0: None}, 
                         315: {.065: None, .1: None, .15: 1, .25: 2, .4: 3, .65: 5, 1.0: 7, 
                             1.5: 10, 2.5: 14, 4.0: 21, 6.5: None, 10.0: None, 15.0: None}, 
                         500: {.065: None, .1: 1, .15: 2, .25: 3, .4: 5, .65: 7, 1.0: 10, 
                             1.5: 14, 2.5: 21, 4.0: None, 6.5: None, 10.0: None, 15.0: None}, 
                         800: {.065: 1, .1: 2, .15: 3, .25: 5, .4: 7, .65: 10, 1.0: 14, 
                             1.5: 21, 2.5: None, 4.0: None, 6.5: None, 10.0: None, 15.0: None}, 
                         1250: {.065: 2, .1: 3, .15: 5, .25: 7, .4: 10, .65:14 , 1.0: 21, 
                             1.5: None, 2.5: None, 4.0: None, 6.5: None, 10.0: None, 15.0: None}, 
                         2000: {.065: 3, .1: 5, .15: 7, .25: 10, .4: 14, .65: 21, 1.0: None, 
                             1.5: None, 2.5: None, 4.0: None, 6.5: None, 10.0: None, 15.0: None}}

    def get_aql_for_batchsize(self, batch_size):
        batch_sizes = [k for k in self.aql_dict.keys()]
        batch_key = min(batch_sizes, key=lambda x: abs (x - batch_size))
        batch_dict = {k: v for k, v in (self.aql_dict[batch_key]).items() if v is not None}
        
        return batch_dict