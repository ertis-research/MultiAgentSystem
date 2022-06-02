class StorageBattery():
    def __init__(self, capacity = 120):
        """
            Initilization method
            : param
                capacity - Capacity of the Storage Battery
        """
        self.capacity = capacity
        self.soc = self.capacity * 0.75
        self.consumption = 0
        
    def charge12(self):
        """
            Simulation of 12Kwh charge (12kwh = 4kw per 15 minutes)
            : return consumption
        """
        c = min(self.capacity, self.soc + 4)
        self.consumption = self.soc - c 
        self.soc = c
        return self.consumption
    
    def charge24(self):
        """
            Simulation of 24Kwh charge (24kwh = 6kw per 15 minutes)
            : return consumption
        """
        c = min(self.capacity, self.soc + 6)
        self.consumption = self.soc - c 
        self.soc = c
        return self.consumption
    
    def charge36(self):
        """
            Simulation of 36Kwh charge (36kwh = 9kw per 15 minutes)
            
        """
        c = min(self.capacity, self.soc + 9)
        self.consumption = self.soc - c 
        self.soc = c
        return self.consumption
    
    def discharge12(self):
        """
            Simulation of 12Kwh discharge (12kwh = 4kw per 15 minutes)
        """
        self.consumption = min(4, self.soc - 4)
        self.soc += self.consumption
        return self.consumption
    
    def discharge24(self):
        """
            Simulation of 24Kwh discharge (24kwh = 6kw per 15 minutes)
            
        """
        self.consumption = min(6, self.soc - 6)
        self.soc += self.consumption
        return self.consumption
        
    
    def discharge36(self):
        """
            Simulation of 36Kwh discharge (36kwh = 9kw per 15 minutes)
            : return consumption
        """
        self.consumption = min(9, self.soc - 9)
        self.soc += self.consumption
        return self.consumption
    
    def stop(self):
        """
            Stop processing
        """
        self.consumption = 0
        
    def render(self, mode="human"):
        """
            Print the ev charging station status
        """
        
        print ("Ampere SB -> Capacity: {}, State of Charge: {}, Consumption: {}".format(self.capacity, self.soc, self.consumption))
    
    def reset(self):
        self.soc = self.capacity 
        self.consumption = 0