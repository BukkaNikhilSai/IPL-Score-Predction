import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



player=pd.read_csv("Player.csv",encoding="ISO-8859-1")
match=pd.read_csv("Match.csv",encoding="ISO-8859-1")
team=pd.read_csv("Team.csv",encoding="ISO-8859-1")
ball_by_ball=pd.read_csv("Ball_By_Ball.csv",encoding="ISO-8859-1")
palayer_match=pd.read_csv("Player_match.csv",encoding="ISO-8859-1")


class Utils(object):

    @staticmethod
    def index(x):
        tokens=x.split(",")
        if(len(tokens[2])==1):
            tokens[2]='0'+tokens[2]
        return ",".join(tokens)

    @staticmethod
    def bowler(x):
        tokens=x.split(",")
        a= ball_by_ball.loc[(ball_by_ball.MatcH_id==int(tokens[0]))&(ball_by_ball.Innings_No==int(tokens[1]))&(ball_by_ball.Over_id==int(tokens[2])),'Bowler']
        return(a[a.index[0]])

    @staticmethod
    def spinners(x):

        spin_off=["Right-arm offbreak"," Right-arm offbreak",
        "Slow left-arm orthodox"]
        spin_leg=["Legbreak googly","Legbreak"," Legbreak", "Slow left-arm chinaman"]

        a=player.loc[player.Player_Id==x,'Bowling_skill']
        if(a[a.index[0]] in spin_off):
            return 1
        elif(a[a.index[0]] in spin_leg):
            return 2
        else :
            return 0

    @staticmethod
    def striker(x):
        ok=x.split(",")
        a= ball_by_ball.loc[(ball_by_ball.MatcH_id==int(ok[0]))&(ball_by_ball.Innings_No==int(ok[1]))&(ball_by_ball.Over_id==int(ok[2])),'Striker']
        return(a[a.index[0]])

    @staticmethod
    def Batting_style(x):
        x=player.loc[(player.Player_Id)==x,'Batting_hand']
        return x[x.index[0]].strip()

    @staticmethod
    def Striker_Batting_Position(x):
        ok=x.split(",")
        a=ball_by_ball.loc[(ball_by_ball.MatcH_id==int(ok[0]))&(ball_by_ball.Innings_No==int(ok[1]))&(ball_by_ball.Over_id==int(ok[2])),'Striker_Batting_Position']
        return(a[a.index[0]])

    @staticmethod
    def PlayerOut(x):
        ok=x.split(",")
        mid= int(ok[0])
        iid=int(ok[1])
        oid=int(ok[2])
        if(oid==1):
            return 0
        else:
            a=ball_by_ball.loc[(ball_by_ball.MatcH_id==mid)&(ball_by_ball.Innings_No==iid)&(ball_by_ball.Over_id==oid-1),'Player_Out']
            count=0
            for i in a:
                if(i!=0):
                    count+=1
            return(count)

    @staticmethod
    def PlayerOut_cur(x):
        ok=x.split(",")
        mid= int(ok[0])
        iid=int(ok[1])
        oid=int(ok[2])
        a=ball_by_ball.loc[(ball_by_ball.MatcH_id==mid)&(ball_by_ball.Innings_No==iid)&(ball_by_ball.Over_id==oid),'Player_Out']
        count=0
        for i in a:
            if(i!=0):
                count+=1
        return(count)

    @staticmethod
    def MatchDateSK(x):
        ok=x.split(",")
        a= ball_by_ball.loc[(ball_by_ball.MatcH_id==int(ok[0]))&(ball_by_ball.Innings_No==int(ok[1]))&(ball_by_ball.Over_id==int(ok[2])),'MatchDateSK']
        return(max(a))

    @staticmethod
    def BattingTeamSK(x):
        ok=x.split(",")
        a= ball_by_ball.loc[(ball_by_ball.MatcH_id==int(ok[0]))&(ball_by_ball.Innings_No==int(ok[1]))&(ball_by_ball.Over_id==int(ok[2])),'BattingTeam_SK']
        return(max(a))

    @staticmethod
    def BowlingTeamSK(x):
        ok=x.split(",")
        a= ball_by_ball.loc[(ball_by_ball.MatcH_id==int(ok[0]))&(ball_by_ball.Innings_No==int(ok[1]))&(ball_by_ball.Over_id==int(ok[2])),'BowlingTeam_SK']
        return(max(a))

    @staticmethod
    def MatchDate(x):
        ok=x.split(",")
        a= ball_by_ball.loc[(ball_by_ball.MatcH_id==int(ok[0]))&(ball_by_ball.Innings_No==int(ok[1]))&(ball_by_ball.Over_id==int(ok[2])),'Match_Date']
        return(max(a))

    @staticmethod
    def innings(x):
        a=x.split(",")
        return int(a[1])

    @staticmethod
    def dob(x):
        try:
            a=player.loc[player["PLAYER_SK"]==x,'DOB']
            f=a[a.index[0]].split("/")
            return int(f[2])
        except:
            return 1999

    @staticmethod
    def year(x):
        a=x.split("/")
        if(a[2]=='2017'):
            return 1
        else :
            return 0

class DataPreprocessing(object):

    def __init__(self):

        self.player=pd.read_csv("Player.csv",encoding="ISO-8859-1")
        self.match=pd.read_csv("Match.csv",encoding="ISO-8859-1")
        self.team=pd.read_csv("Team.csv",encoding="ISO-8859-1")
        self.ball_by_ball=pd.read_csv("Ball_By_Ball.csv",encoding="ISO-8859-1")


    def preprocess(self):

        #Fill the null values
        self.ball_by_ball.Striker_Batting_Position.fillna(4,inplace=True)
        self.ball_by_ball.Player_Out.fillna(value=0, inplace=True)
        ball_by_ball.Striker_Batting_Position.fillna(4,inplace=True)
        ball_by_ball.Player_Out.fillna(value=0, inplace=True)
        features = ["MatcH_id","Over_id","Innings_No","Runs_Scored","Extra_runs"]

        data = self.ball_by_ball[features]

        #Gruoping into overs
        data['id'] = data['MatcH_id'].apply(str)+","+data['Innings_No'].apply(str)+","+data['Over_id'].apply(str)
        data["id"] = data.id.apply(Utils.index)
        data = pd.DataFrame(data[['id','Runs_Scored',"Extra_runs"]].groupby('id').sum())

        print("Creating bowler features")
        #Getting bowler DataFrame
        data['id'] = data.index
        data["Bowler"]=data.id.apply(lambda x:Utils.bowler(x))
        data["BowlerType"]=data.Bowler.apply(Utils.spinners)
        data["Striker"]=data.id.apply(lambda x:Utils.striker(x))
        data["wickets"]=data.id.apply(lambda x:Utils.PlayerOut_cur(x))

        bowl_skill = data.groupby(['Bowler', 'wickets'])['Bowler'].count().unstack('wickets').fillna(0)
        bowl_skill["total"]=bowl_skill[1]+bowl_skill[2]+bowl_skill[3]+bowl_skill[4]
        bowl_skill["sum"]=bowl_skill[0]+bowl_skill[1]+bowl_skill[2]+bowl_skill[3]+bowl_skill[4]
        bowl_skill['Bowler'] = bowl_skill.index
        dict1=bowl_skill.to_dict()

        data['bowl_skill']=data['Bowler'].apply(lambda x:dict1['total'][x])
        data['bowl_experience']=data['Bowler'].apply(lambda x:dict1['sum'][x])

        data["bowler_dob"]=data.Bowler.apply(Utils.dob)


        #Generating label
        data['total_runs']=data['Runs_Scored']+data['Extra_runs']
        data['high_runs']=data['total_runs']>=12


        print("Creating batsmen features")
        #getting batsmen data
        data["batting_hand"]=data.Striker.apply(Utils.Batting_style)
        le=LabelEncoder()
        data["batting_hand"]=le.fit_transform(data["batting_hand"])
        data["strike_position"]=data.id.apply(lambda x:Utils.Striker_Batting_Position(x))

        bat_skill = data.groupby(['Striker', 'high_runs'])['Striker'].count().unstack('high_runs').fillna(0)
        bat_skill["sum"]=bat_skill[True]+bat_skill[False]
        bat_skill['Striker'] = bat_skill.index
        dic=bat_skill.to_dict()

        data['bat_skill']=data['Striker'].apply(lambda x:dic[True][x])
        data['bat_Experience']=data['Striker'].apply(lambda x:dic["sum"][x])

        data["striker_dob"]=data.Striker.apply(Utils.dob)


        print("Creating match data")
        #match data
        data["playerout"]=data.id.apply(lambda x:Utils.PlayerOut(x))
        data['matchdatesk']=data.id.apply(lambda x:Utils.MatchDateSK(x))
        data["matchdate"]=data.id.apply(lambda x:Utils.MatchDate(x))
        data["innings"]=data.id.apply(Utils.innings)
        data["matchyear"]=data["matchdate"].apply(lambda x:x.split("/"))
        data["matchyear"]=data["matchyear"].apply(lambda x :x[2])
        data["over"]=data.id.apply(lambda x:int(x.split(",")[2]))
        data["striker_age"]=data["matchyear"].apply(int)-data["striker_dob"]
        data["bowler_age"]=data["matchyear"].apply(int)-data["bowler_dob"]

        print("Creating team data")
        #team data
        data['bowlingteamsk']=data.id.apply(lambda x:Utils.BowlingTeamSK(x))
        data['battingteamsk']=data.id.apply(lambda x:Utils.BattingTeamSK(x))


        print("Creating current match related data")
        #current match related columns
        data["cumulative_overs"]=0
        def cumulative(x):
            a=x.split(",")
            mth=int(a[0])
            inid=int(a[1])
            ovid=int(a[2])
            if(ovid==1):
                return 0
            else:
                if(len(str(ovid-1))==1):
                    f=str(mth)+","+str(inid)+","+"0"+str(ovid-1)
                else:
                    f=str(mth)+","+str(inid)+","+str(ovid-1)
                aa=data.loc[data.id==f,'cumulative_overs'].values[0]
                if(data[data.id==x].high_runs.bool()):
                    return (aa+1)
                else:
                    return aa
        data=data.sort_values(by=['id'])
        data["cumulative_overs"]=data.id.apply(cumulative)

        data["bowl_skill"]=data["bowl_skill"]/data["bowl_experience"]
        data["bat_skill"]=data["bat_skill"]/data["bat_Experience"]

        #seperating test data
        def year(x):
            a=x.split("/")
            if(a[2]=='2017'):
                return 1
            else :
                return 0
        data["is2017"]=data.matchdate.apply(lambda x:year(x))

        data = shuffle(data)

        data.rename(columns={'strike_position':'Batsman_position',
                   'bat_skill':'Batsmen_skill',
                   'bat_Experience':'Batsman_Experience',
                    'bowl_skill':'Bowler_skill',
                     'bowl_experience':'Bowler_experience',
                      "striker_age":"Batsman_age",
                  "bowler_age":"Bowler_age"       ,
                    "cumulative_overs":"Previous_Highscoring_Overs",
                   "playerout":"Previous_wickets" } ,
                 inplace=True)
        return data


class Model(object):

    def __init__(self):

        Data = DataPreprocessing()
        self.data = Data.preprocess()

    def train_test_split(self):

        self.train=self.data[self.data.is2017==0]
        self.test=self.data[self.data.is2017==1]

    def train_model(self):
        feats=["Batsman_position","over","Batsmen_skill","Batsman_Experience",'Bowler_skill','Bowler_experience','innings',
             "BowlerType","Batsman_age","Bowler_age","Previous_Highscoring_Overs","Previous_wickets"]

        X = self.train[feats]
        y = self.train["high_runs"]

        self.clf = xgb.XGBClassifier()
        self.clf.fit(X, y)

    def test_model(_self):
        x_val = self.test[num_feats]
        y_val = self.test['high_runs']
        y_pred_xgb = self.clf.predict(x_val)
        return accuracy_score(y_val,y_pred_xgb)

m = Model()
m.train_test_split()
m.train_model()
print("Accuracy Score:: " ,m.test_model())
