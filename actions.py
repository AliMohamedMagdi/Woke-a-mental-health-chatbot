from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from logging import error
from typing import Any, Text, Dict, List
from pymongo import MongoClient
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from . import learning
from . import mood
class diagnose(Action):

    def name(self) -> Text:
        return "diagnose"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        age = tracker.get_slot("age")
        gender = tracker.get_slot("gender")
        self_employment = tracker.get_slot("self_employment")
        family_history = tracker.get_slot("family_history")
        treatment = tracker.get_slot("treatment")
        work_interference = tracker.get_slot("work_interference")
        tech_company = tracker.get_slot("tech_company")
        benefits= tracker.get_slot("benefits")
        care_options= tracker.get_slot("care_options")
        anonymity= tracker.get_slot("anonymity")
        medical_leave= tracker.get_slot("medical_leave")
        willing_to_discuss_collegaue= tracker.get_slot("willing_to_discuss_collegaue")
        if(self_employment is None):
         self_employment="          "
        if(family_history is None):
         family_history="          "
        if(treatment is None):
         treatment="          "
        if(work_interference is None):
         work_interference="          "
        if(tech_company is None):
         tech_company="          "
        if(benefits is None):
         benefits="          "
        if(care_options is None):
         care_options="          "
        if(anonymity is None):
         anonymity="          "
        if(medical_leave is None):
         medical_leave="          "
        if(willing_to_discuss_collegaue is None):
         willing_to_discuss_collegaue="          " 
       
        ##validation
        if(gender[0]=="m"or gender[0]=="M"):
            gender="Male"
        elif(gender[0]=="f" or gender[0]=="F"):
         gender="Female"
        else:
         if (gender!="          "):
           gender="Trans"

        if(self_employment[0]=="Y"or self_employment[0]=="y"):
            self_employment="Yes"
        elif(self_employment[0]=="N" or self_employment[0]=="n"):
         self_employment="No"

        if(family_history[0]=="Y"or family_history[0]=="y"):
            family_history="Yes"
        elif(family_history[0]=="N" or family_history[0]=="n"):
         family_history="No"

        if(treatment[0]=="Y"or treatment[0]=="y"):
            treatment="Yes"
        elif(treatment[0]=="N" or treatment[0]=="n"):
         treatment="No"

        if(work_interference[0]=="o"or work_interference[0]=="O"):
            work_interference="Often"
        elif(work_interference[0]=="s" or work_interference[0]=="S"):
         work_interference="Sometimes"
        elif(work_interference[0]=="N" or work_interference[0]=="n"):
         work_interference="Never"
        elif(work_interference[0]=="R" or work_interference[0]=="r"):
         work_interference="Rarely"

        if(tech_company[0]=="Y"or tech_company[0]=="y"):
            tech_company="Yes"
        elif(tech_company[0]=="N" or tech_company[0]=="n"):
         tech_company="No"

        if(benefits[0]=="Y"or benefits[0]=="y"):
            benefits="Yes"
        elif(benefits[0]=="N" or benefits[0]=="n"):
         benefits="No"
        elif(benefits[0]=="D" or benefits[0]=="d"):
         benefits="Don't know" 

        if(care_options[0]=="Y"or care_options[0]=="y"):
            care_options="Yes"
        elif(care_options[0]=="N" and care_options[1]=="o" or care_options[0]=="n" and care_options[1]=="o"):
         care_options="No"
        else:
          if (care_options!="          "):
           care_options="Not sure"    

        if(anonymity[0]=="Y"or anonymity[0]=="y"):
            anonymity="Yes"
        elif(anonymity[0]=="N" or anonymity[0]=="n"):
         anonymity="No"
        else:
         if (anonymity!="          "): 
          anonymity="Don't know"

        if(medical_leave[0]=="e"or medical_leave[0]=="E"):
            medical_leave="easy"
        elif(medical_leave[0]=="D" or medical_leave[0]=="d"):
         medical_leave="difficult"
        elif(medical_leave[0]=="S" or medical_leave[0]=="s"):
         medical_leave="smooth"
        elif(medical_leave[0]=="c" or medical_leave[0]=="C"):
         medical_leave="challenging"
        else:
         if (medical_leave!="          "):
          medical_leave="Don't know"
        

        if(willing_to_discuss_collegaue[0]=="Y"or willing_to_discuss_collegaue[0]=="y"):
            willing_to_discuss_collegaue="Yes"
        elif(willing_to_discuss_collegaue[0]=="N" or willing_to_discuss_collegaue[0]=="n"):
         willing_to_discuss_collegaue="No"
        else:
         if (willing_to_discuss_collegaue!="          "):
          willing_to_discuss_collegaue="Some of them"
        new_input=["      ",age,gender,"      ","      ",self_employment,family_history,treatment,work_interference,"      ","      ",tech_company,benefits,care_options,"      ","      ",anonymity,medical_leave,"      ","      ",willing_to_discuss_collegaue,"      ","      ","      ","      ","      ","      "]
        print(new_input)
        learning.append_new(new_input)
        dec=learning.process_predict()
        print(dec)
        if(dec==0):
         dispatcher.utter_message("no immediate treatment required")
        else:
         dispatcher.utter_message("immediate treatment required")
        
        return

class isDepressed(Action):

    def name(self) -> Text:
        return "is_depressed"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        sym1 = tracker.get_slot("depression_lethargy")
        sym2 = tracker.get_slot("depression_apathy")
        sym3 = tracker.get_slot("depression_sleep_disorder")
        sym4 = tracker.get_slot("depression_anger issues")
        sym5 = tracker.get_slot("depression_loss of energy")
        sym6 = tracker.get_slot("depression_self_hate")
        if (sym1 == True and sym2==True and sym3==True and sym4==True and sym5==True and sym6==True):
         dispatcher.utter_message(text="you have all the depression symptoms you might have depression you should seek therapy")
        return
        
class email(Action):

    def name(self) -> Text:
        return "email"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        import smtplib, ssl

        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "rosaoscura11@gmail.com"  
        receiver_email = "rosaoscura11@gmail.com" 
        password = "rosaoscura11@gmail.comrosaoscura11@gmail.comrosaoscura11@gmail.com"
        message = """\
        Subject: Warning

        Warning."""

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        return

class timeimprovment(Action):
    
    def name(self) -> Text:
        return "time"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        timeimp=False
        def flatten_json(y):
            out = {}

            def flatten(x, name=''):
                if type(x) is dict:
                   for a in x:
                      flatten(x[a], name + a + '_')
                elif type(x) is list:
                     i = 0
                     for a in x:
                      flatten(a, name + str(i) + '_')
                      i += 1
                else:
                    out[name[:-1]] = x

            flatten(y)
            return out

        def calc_avg(interaction_number):
         client = MongoClient('mongodb://localhost:27017')
         mydb = client["woke"]
         mycol = mydb["conversations"]
         my_list = []
         avr=0
         email = tracker.get_slot("email")
         myquery = { "slots.email": email,"slots.count":interaction_number}

         for x in mycol.find(myquery,{ "_id":0 ,"events.timestamp": 1,"events.input_channel": 1}):
          my_list.append(x)

         flat=flatten_json(my_list)

 
         totaltime=list(flat.values())
         count=len(totaltime)
         ts=[]
         j=0
         from datetime import datetime
         for i in range(count-1):
          if(i+1<count):
           if(totaltime[i+1]=="cmdline" or totaltime[i+1]=="rest" ):
            ts.append(0)
            item = datetime.fromtimestamp( float(totaltime[i]) )
              ##converting the time stamp
            ts[j]=item 
            j=j+1
         diff=[]
         count=len(ts)
         j=0
 
         for i in range(count-1):
          if(i+1<count):
           diff.append(0)
           diff[j]=ts[i+1]-ts[i]
           j=j+1
 
         avg=0
         n=0 
         for x in diff:
          avg=avg+x.total_seconds()
          n=n+1
          avr=avg/n
 
         return avr 

        def ci(email):
         client = MongoClient('mongodb://localhost:27017')
         mydb = client["woke"]
         mycol = mydb["conversations"]
         my_list = []
         avr=0
         myquery = { "slots.email": email}
         count=0
         for x in mycol.find(myquery):
          count=count+1
         return count
        email = tracker.get_slot("email")
        counter=ci(email)
        if(counter>=1):
         counter=counter-1
         aver=[]
         j=0
         for i in range(counter, counter-3, -1):
          aver.append(0)
          aver[j]=calc_avg(i)
          j=j+1
         if(aver[0]!=0 and aver[1]!=0):
          if(aver[0]<aver[1]):
           timeimp=True ##time is improved
          else:
           timeimp=False ## time is not improved
         print(aver)
         print (timeimp)
        return [SlotSet("time", timeimp)]

class RememberUser(Action):
    def name(self) -> Text:
        return "Remember_User"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        def flatten_json(y):
            out = {}
            def flatten(x, name=''):
             if type(x) is dict:
                for a in x:
                 flatten(x[a], name + a + '_')
             elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + '_')
                    i += 1
             else:
                 out[name[:-1]] = x

            flatten(y)
            return out
        client = MongoClient('mongodb://localhost:27017')
        mydb = client["woke"]
        mycol = mydb["conversations"]
        email = tracker.get_slot("email")
        name = tracker.get_slot("name")
        f=0
        n=0
        mq={"slots.email": email}
        mydoc = mycol.find(mq)
        for x in mydoc:
         n=n+1
        myquery = { "slots.email": email}
        mydoc = mycol.find(myquery)
        if(n>1):
         for x in mydoc:
          dispatcher.utter_message("I Remember our last conversation do you like to chat with me " + name)
          f = f +1
          break
        if(f==0):
         dispatcher.utter_message("Hi " + name + " it seems this is the first time we speak do you like to chat with me")
        feel =str(tracker.latest_message['text'])
        pfeel=mood.predict(feel)
        return[SlotSet("count", n), SlotSet("feeling", pfeel)]




class empathy(Action):
    def name(self) -> Text:
        return "check_feeling"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
         feel=tracker.get_slot("feeling")
         if(feel=="sadness"):
          dispatcher.utter_message("You cannot protect yourself from sadness without protecting yourself from happiness")
         elif(feel=="anger"):
          dispatcher.utter_message("Never do anything when you are in a temper, for you will do everything wrong")
         elif(feel=="joy"):
          dispatcher.utter_message("When you do things from your soul, you feel a river moving in you, a joy")
         elif(feel=="fear"):
          dispatcher.utter_message("Each of us must confront our own fears, must come face to face with them. How we handle our fears will determine where we go with the rest of our lives. To experience adventure or to be limited by the fear of it")
         elif(feel=="love"):
          dispatcher.utter_message("If I know what love is, it is because of you")     
            
         return
