# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:07:37 2016

@author: Shaw
"""
import pandas
import scipy
import pyomo
import pyomo.opt
import pyomo.environ as pe
import logging
import networkx
import geoplotter ##please use my changed version of geoplotter
import matplotlib.pylab as pylab


class BuildNetwork():
    def __init__(self, fname='austin.csv'):
        self.df = pandas.read_csv(fname)
        self.add = pandas.read_csv('addresses.csv')
        self.plotter = geoplotter.GeoPlotter()
        self.net = networkx.DiGraph(node_styles=dict(default=dict()),
                                    edge_styles=dict(default=dict(color='blue', linewidths=0.1)))
        self.nodearray = []
        self.nodeframe = pandas.DataFrame(columns=['node'])
        self.E = []

    def getstart(self):  ##get float list of the start point coordinates
        self.start = self.df.kmlgeometry.str.extract('\(([0-9-.]* [0-9-.]*),')
        self.start = self.start.to_frame()
        self.start = pandas.DataFrame(self.start.kmlgeometry.str.split(' ', 1).tolist(), columns=['x', 'y'])
        self.start.x = self.start.x.astype(float)
        self.start.y = self.start.y.astype(float)
        self.df['Start'] = self.start.x.astype(str) + ' ' + self.start.y.astype(str)
        self.startlist = list(zip(self.start.x, self.start.y))  ##list of start points (x,y)

    def getend(self): ##get float list of the end point coordinates
        self.end = self.df.kmlgeometry.str.extract(',([0-9-.]* [0-9-.]*)\)')
        self.end = self.end.to_frame()
        self.end = pandas.DataFrame(self.end.kmlgeometry.str.split(' ', 1).tolist(), columns=['x', 'y'])
        self.endstring = self.end
        self.end.x = self.end.x.astype(float)
        self.end.y = self.end.y.astype(float)
        self.df['End'] = self.end.x.astype(str) + ' ' + self.end.y.astype(str)
        self.endlist = list(zip(self.end.x, self.end.y))  ##list of end points (x,y)

    def create_net(self):
        """Draws a network on the map.

        net should have the following attributes:
            node_styles -- a dictionary of style dictionaries for nodes, including a 'default'
            edge_styles -- a dictionary of style dictionaries for edges, including a 'default'

        Each node/edge data dictionary has a 'style' entry that specifies the style to be looked up
        in node_styles/edge_styles.  If no style is specified the default style is used.  Only attributes
        of the default style are changed in plotting."""
        for i in scipy.arange(len(self.start.x)):
            node1 = self.df.Start[i]  ##shoter version of decimal by string conversion for node name
            node2 = self.df.End[i]
            self.net.add_node(node1, lon=self.start.x[i], lat=self.start.y[i], style='default')
            self.net.add_node(node2, lon=self.end.x[i], lat=self.end.y[i], style='default')
            if self.df.ONE_WAY[i] == 'FT':
                self.net.add_edge(node1, node2, style='default', time=self.df.SECONDS[i])
                self.nodearray.append([self.start.x[i], self.start.y[i]])
            elif self.df.ONE_WAY[i] == 'TF':
                self.net.add_edge(node2, node1, style='default', time=self.df.SECONDS[i])
                self.nodearray.append([self.start.x[i], self.start.y[i]])
            else:
                self.net.add_edge(node1, node2, style='default', time=self.df.SECONDS[i])
                self.net.add_edge(node2, node1, style='default', time=self.df.SECONDS[i])
                self.nodearray.append([self.start.x[i], self.start.y[i]])
                self.nodearray.append([self.start.x[i], self.start.y[i]])
        self.nodearray = scipy.array(self.nodearray)

    def create_frame(self):## data frame for Pyomo use
        self.nodearray = []
        arclist = []
        for i in scipy.arange(len(self.start.x)):
            node1 = self.df.Start[i]  ##shoter version of decimal by string conversion for node name
            node2 = self.df.End[i]
            if self.df.ONE_WAY[i] == 'FT':
                arclist.append([node1, node2, self.df.SECONDS[i]])  ##list for recording nodes used pyomo
                self.nodearray.append([self.start.x[i], self.start.y[i]])
            elif self.df.ONE_WAY[i] == 'TF':
                arclist.append([node2, node1, self.df.SECONDS[i]])
                self.nodearray.append([self.start.x[i], self.start.y[i]])
            else:
                arclist.append([node1, node2, self.df.SECONDS[i]])
                arclist.append([node2, node1, self.df.SECONDS[i]])
                self.nodearray.append([self.start.x[i], self.start.y[i]])
                self.nodearray.append([self.start.x[i], self.start.y[i]])
        self.arcframe = pandas.DataFrame(arclist, columns=['Start', 'End', 'Time'])
        self.arcframe = self.arcframe.sort(['Time'], ascending=[1])
        self.arcframe = self.arcframe.drop_duplicates(subset=['Start', 'End'], keep='first')
        self.arcframe.set_index(['Start', 'End'], inplace=True)  ##dataframe of arcs for pyomo

    def drawstreets(self):##Draw the big map
        self.plotter.drawNetwork(self.net)
        self.plotter.setZoom(-97.8526, 30.2147, -97.6264, 30.4323)
        self.plotter.getAxes().get_xaxis().set_visible(False)
        self.plotter.getAxes().get_yaxis().set_visible(False)

    def drawaddresses(self):## Drop attaction points on the map
        for i in scipy.arange(len(self.add.Lat)):
            if self.add.Address[i] == 'Engineering Teaching Center, 304 E 26 1/2 St, Austin, TX':
                self.plotter.drawPoints(self.add.Lon[i], self.add.Lat[i], color='green', s=1)
            else:
                self.plotter.drawPoints(self.add.Lon[i], self.add.Lat[i], color='red', s=1)

    def findclosest(self):  ##data is a dictionary of the coordinates
        self.add['Closest_Node'] = self.add['Lon']
        self.addressarray = []
        for i in scipy.arange(len(self.add.Lon)):
            self.addressarray.append([self.add.Lon[i], self.add.Lat[i]])
        self.addressarray = scipy.array(self.addressarray)
        i = 0
        for loc in self.addressarray:
            dis = -scipy.sqrt(scipy.sum((self.nodearray - loc) * (self.nodearray - loc), axis=1))
            idx = scipy.argmax(dis)
            node = str(self.nodearray[idx][0]) + ' ' + str(self.nodearray[idx][1])
            self.add.Closest_Node.iloc[i] = node
            i += 1

    def getSPNetworkx(self, id1, id2):##Networkx solution
        start = self.add.Closest_Node.loc[id1]
        end = self.add.Closest_Node.loc[id2]
        route = networkx.shortest_path(self.net, source=start, target=end, weight='time')
        return route
    def getSPNetworkxtime(self, id1, id2):##Networkx solution
        start = self.add.Closest_Node.loc[id1]
        end = self.add.Closest_Node.loc[id2]
        time = networkx.shortest_path_length(self.net, source=start, target=end, weight='time')
        return time

    # def drawroute(self, route,zoom1,zoom2,zoom3,zoom4):##draw a path from the networkx result
    #     routel = len(route)
    #     self.plotter.setZoom(zoom1,zoom2,zoom3,zoom4)
    #     for i in scipy.arange(routel - 1):
    #         node1 = route[i]
    #         node2 = route[i + 1]
    #         way = self.df.kmlgeometry[((self.df.Start == node1) & (self.df.End == node2)) | (
    #         (self.df.Start == node2) & (self.df.End == node1))].str.extract('\((.*)\)').tolist()
    #         #            if not self.df[((self.df.Start==node2) & (self.df.End == node1))].empty:
    #         #                if self.df[((self.df.Start==node1) & (self.df.End == node2))].empty:
    #         #                    print node1, node2
    #         #                    print self.df[self.df.End==node1].ONE_WAY
    #
    #         way = way[0].split(',')
    #         self.drawway(way)
    #
    # def drawway(self, way):##draw a single steet path from data
    #     line = []
    #     r = []
    #     waylen = len(way)
    #     for i in scipy.arange(waylen):
    #         node1 = way[i]
    #         node1 = node1.split()
    #         node1 = [float(i) for i in node1]
    #         r.append((node1[0], node1[1]))
    #     line.append(r)
    #     self.plotter.drawLines(line, color='orange', alpha=0.92)
    #
    # def getSPCplex(self):##Pyomo solution takes about 40 min to get result
    #     self.nodeframe = pandas.DataFrame(self.net.nodes(), columns=['node'])  ##dataframe of nodes
    #     self.nodeframe['Imbalance'] = 0
    #     self.nodeframe.set_index(['node'], inplace=True)
    #     self.nodeset = self.nodeframe.index.unique()
    #     self.arcset = self.arcframe.index.unique()
    #     start = self.add.Closest_Node.loc[15]
    #     end = self.add.Closest_Node.loc[3]
    #     self.nodeframe.Imbalance[start] = 1
    #     self.nodeframe.Imbalance[end] = -1
    #     print self.nodeframe.Imbalance
    #     path = self.createModel()
    #     print path
    #
    # def createModel(self):##Pyomo model
    #     self.m = pe.ConcreteModel()
    #     self.m.node_set = pe.Set(initialize=self.nodeset)
    #     self.m.arc_set = pe.Set(initialize=self.arcset, dimen=2)
    #     self.m.Y = pe.Var(self.m.arc_set, domain=pe.NonNegativeReals)
    #
    #     def obj_rule(m):
    #         return sum(m.Y[e] * self.arcframe.ix[e, 'Time'] for e in self.arcset)
    #
    #     self.m.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)
    #
    #     def flow_bal_rule(m, n):
    #         arcs = self.arcframe.reset_index()
    #         preds = arcs[arcs.End == n]['Start']
    #         succs = arcs[arcs.Start == n]['End']
    #         return sum(m.Y[(p, n)] for p in preds) - sum(m.Y[(n, s)] for s in succs) == self.nodeframe.ix[
    #             n, 'Imbalance']
    #
    #     self.m.FlowBal = pe.Constraint(self.m.node_set, rule=flow_bal_rule)
    #     solver = pyomo.opt.SolverFactory('cplex')
    #     results = solver.solve(self.m, tee=True, keepfiles=False,
    #                            options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")
    #     if results.solver.status != pyomo.opt.SolverStatus.ok:
    #         logging.warning('Check solver not ok?')
    #     if results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal:
    #         logging.warning('Check solver optimality?')
    #     return results
    #
    # def drawpyomo(self,zoom1,zoom2,zoom3,zoom4):##draw pyomo result
    #     path = []
    #     for i in self.m.Y.keys():
    #         if self.m.Y[i] == 1:
    #             path.extend(list(i))
    #     self.drawstreets()
    #     self.drawaddresses()
    #     self.drawroute(path,zoom1,zoom2,zoom3,zoom4)
    #     pylab.show()
    def get_paths(self):
        dict = {}
        # path = {}
        # times = {}
        range = len(self.add.Lon)
        for i in scipy.arange(range):
            for j in scipy.arange(range):
                if i != j:
                    dict[(self.add.ShortHand.iloc[i], self.add.ShortHand.iloc[j])] = {}
                    dict[(self.add.ShortHand.iloc[i], self.add.ShortHand.iloc[j])]['path'] = self.getSPNetworkx(i,j)
                    a = float(self.getSPNetworkxtime(i, j))
                    dict[(self.add.ShortHand.iloc[i], self.add.ShortHand.iloc[j])]['time'] = a/3600
                    dict[(self.add.ShortHand.iloc[i], self.add.ShortHand.iloc[j])]['load'] = float(self.add.LoadingTime.iloc[j])
                    dict[(self.add.ShortHand.iloc[i], self.add.ShortHand.iloc[j])]['value'] = -float(self.add.Value.iloc[j])
                    dict[(self.add.ShortHand.iloc[i], self.add.ShortHand.iloc[j])]['start'] = float(self.add.TimeWindowStart.iloc[j])
                    dict[(self.add.ShortHand.iloc[i], self.add.ShortHand.iloc[j])]['end'] = float(self.add.TimeWindowEnd.iloc[j])
                else:
                    continue

        self.data = pandas.DataFrame.from_dict(dict,orient='index')
        self.data=self.data.reset_index()
        return self.data

    def create_dict(self):
        import pickle
        output = open('pathtime.txt','r')
        self.data = pickle.load(output)
        output.close();
        self.step_dict = {}
        for i in self.add.ShortHand:
            self.step_dict[i] = []
        return self.step_dict
    def start_route(self,startpoint,starttime,maxvol):
        self.step_dict[startpoint].append([starttime,maxvol,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) ## the alpha in the algorithm
        self.E.append(startpoint) ## points wait to be treated
        self.adds = self.add.ShortHand.tolist() ##list of locations

    def algorithm(self,endtime):
        import random
        while self.E:
            vi=random.choice(self.E)
            succ=list(self.add.ShortHand).remove(vi)
            for vj in succ:
                j = self.adds.index(vj)
                trans_time = self.data.time[self.data.level_0 == vi and self.data.level_1 == vj]
                dest_load = self.data.load[self.data.level_0 == vi and self.data.level_1 == vj]
                dest_reven = self.data.value[self.data.level_0 == vi and self.data.level_1 == vj]
                F = []
                for numta in self.dict[vi]:
                    if numta[j]  == 0:
                        F.append(self.reachability(numta,vi,vj,endtime))
                combined  = F.append(self.dict[vj])##very important step, generate next lable, including test reachbability of other nodes
                self.domination(combined)
                if self.dict[vj] != combined:
                    self.dict[vj] = combined
                    self.E.append(vj)
            self.E.remove(vi)



    def domination(self,inputlist):
        from operator import itemgetter
        lists = inputlist
        sort_list = sorted(lists,key=itemgetter(2))
        var=12 / 0
        numtas = len(lists)
        j=0
        while j < numtas-1:
            numta1 = sort_list[j]##first numta
            numta2 = sort_list[j+1]##second numta
            rech1 = numta1[3:19]
            rech2 = numta2[3:19]
            match = 0
            if numta1[2]<=numta2[2]: ##compare s
                if ((numta1[0]<=numta2[0]) and (numta1[1]<=numta2[1]) and (rech1<=rech2) and (numta1[19]<=numta2[19])):
                    sort_list.remove(numta2)
                    numtas = len(sort_list)
                else:
                    j=j+1
            else:
                j=j+1
        return sort_list



    def reachability(self,numta,vi,vj,endtime):
        trans_time = self.data.time[self.data.level_0 == vi and self.data.level_1 == vj]
        dest_load = self.data.load[self.data.level_0 == vi and self.data.level_1 == vj]
        dest_reven = self.data.value[self.data.level_0 == vi and self.data.level_1 == vj]
        j =self.add.ShortHand.tolist().index(vj)
        F = []
        F.append(trans_time)
        F.append(-dest_load)
        F.append(1)
        temp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        temp[j] = 1
        F.append(temp)
        F.append(dest_reven)
        new_numta = numta+F
        reach_vec = new_numta[3:19]
        for i in scipy.range(reach_vec):
            if reach_vec[i] == 0:
                if (new_numta[0]+self.data.time[self.data.level_0==vj and self.data.level_1==self.adds[i]])<=endtime:
                    if (new_numta[0]+self.data.time[self.data.level_0==vj and self.data.level_1==self.adds[i]]<=self.add.TimeWindowEnd[self.add.ShortHand == self.adds[i]]):
                        continue
                    else:
                        reach_vec[i] = 1
                else:
                    reach_vec[i] = 1
        new_numta[3:19] = reach_vec
        return new_numta









if __name__ == '__main__':
    BN = BuildNetwork();
    # BN.getstart();
    # BN.getend();
    ##Solution from Networkx
    # BN.create_net();
    # BN.findclosest();
    BN.create_dict();
    BN.start_route('ETC',9,50)

    # net  = BN.net;
    # BN.get_paths();

    # path = BN.getSPNetworkx(15, 3)
    # BN.drawstreets();
    # BN.drawaddresses();
    # BN.drawroute(path,-97.8526, 30.2147, -97.6264, 30.4323)  #
    # pylab.show()
    ## Run Pyomo CAUTION:: This part will take significantly amount of time
    # BN.create_frame();
    # BN.getSPCplex();
    # BN.drawpyomo(-97.8526, 30.2147, -97.6264, 30.4323)