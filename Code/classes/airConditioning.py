class Airconditioning():
    def __init__(self):
        self.state = 0
        
    def turnOff(self):
        self.state = 0
        return self.consumption()
         
    def smallChange(self):
        self.state = 1
        return self.consumption()
    
    def bigChange(self):
        self.state = 2
        return self.consumption()
        
    def render(self, mode="human"):
        print("Air conditioning -> State {}, Consumption {}".format(self.state, self.consumption()))
    
    def consumption(self):
        if self.state == 0:
            return -20.0
        else:

            return -round(20.0 + self.state*20.0, 2)