import aiwolfpy
import aiwolfpy.contentbuilder as cb
import random
import optparse
import sys

# Dummy agent that shows its predictions
# Based on https://github.com/ehauckdo/AIWoof
# If team VILLAGER targets the werewolf with more probability
# If team WEREWOLF tries to target the SEER and the MEDIUM

class conan(object):

    def __init__(self, agent_name):
        self.myname = agent_name

    def initialize(self, base_info, diff_data, game_setting):
        self.id = base_info["agentIdx"]
        self.base_info = base_info
        self.game_setting = game_setting

        self.game_history = {}  # dict to store the sentences stated each day
        self.player_map = {}  # dict with all the information inferred of each player
        self.arguments = [] # list of arguments (BECAUSE) stated
        self.revealed = {} # dict with comeout roles
        self.role_vector = [] # list of players most probable roles
        self.role = base_info["myRole"]
        self.current_target = None

        self.updatePlayerMap(base_info)

    def getName(self):
        return self.myname

    def update(self, base_info, diff_data, request):
        print("Executing update...")
        self.base_info = base_info

        self.processMessages(diff_data)
        self.updatePlayerMap(base_info)

    def dayStart(self):
        print("Day start: ")

    def talk(self):
        print("Talking phase: ")

    def whisper(self):
        print("Whispering: ")

    def vote(self):
        print("Voting: ")

    def attack(self):
        print("Attacking: ")

    def divine(self):
        print("Making a divination: ")

    def guard(self):
        print("Guarding: ")

    def finish(self):
        print("Good game!")

    def updatePlayerMap(self, base_info):
        if self.player_map == None:
            self.player_map = {}

        for key, value in base_info["statusMap"].items():
            agent_id = int(key)
            if agent_id is not self.id:
                if agent_id not in self.player_map:
                    self.player_map[agent_id] = {}
                    self.player_map[agent_id]["lies"] = []
                    self.player_map[agent_id]["fake_lies"] = []
                    self.player_map[agent_id]["dangerous_agent"] = False
                self.player_map[agent_id]["status"] = value


        print(self.player_map)

    def processMessages(self, diff_data):
        for row in diff_data.itertuples():

            agent = getattr(row, "agent")
            text = getattr(row, "text")

            # Añado argumentos a la lista de argumentos
            if "BECAUSE" in text:
                self.arguments.append([agent, text])

            # Almaceno los roles revelados y si revela distintos roles almaceno la mentira
            if "COMINGOUT" in text:
                t = text.split()
                if len(t) == 3:
                    if "co_role" not in self.player_map[agent]:
                        self.player_map[agent]["co_role"] = t[2]
                    else:
                        self.player_map[agent]["lies"].append([agent, text])

            # Si estiman mi rol incorrectamente almaceno la mentira
            if "ESTIMATE" in text:
                t = text.split()
                if len(t) == 3:
                    if ("{:02d}".format(self.id) in t[1]) and (self.role != t[2]):
                        self.player_map[agent]["lies"].append([agent, text])

            # Analizar el texto buscando incongruencias
            # 15: 1 seer, medium, possessed, bodyguard, 8 villa, 3 ww
            # 5: 1 seer, possessed, ww, 2 villa

            # Si soy ROL_UNICO y alguien se revela como ROL_UNICO lo añado a la lista de mentirosos
            if self.role == "SEER":
                if "SEER" in text and "COMINGOUT" in text:
                    self.player_map[agent]["lies"].append([agent, text])

            elif self.role == "MEDIUM":
                if "MEDIUM" in text and "COMINGOUT" in text:
                    self.player_map[agent]["lies"].append([agent, text])

            elif self.role == "BODYGUARD":
                if "BODYGUARD" in text and "COMINGOUT" in text:
                    self.player_map[agent]["lies"].append([agent, text])

            # Si soy del equipo WW y han acertado mi rol, me guardo el mensaje para construir argumentos en contra
            elif self.role == "POSSESSED":
                if ("{:02d}".format(self.id) in text) and ("POSSESSED" in text or "VOTE" in text):
                    self.player_map[agent]["dangerous_agent"] = True
                    if "ESTIMATE" in text:
                        self.player_map[agent]["fake_lies"].append([agent, text])

            elif self.role == "WEREWOLF":
                if ("{:02d}".format(self.id) in text) and ("WEREWOLF" in text or "VOTE" in text):
                    self.player_map[agent]["dangerous_agent"] = True
                    if "ESTIMATE" in text:
                        self.player_map[agent]["fake_lies"].append([agent, text])



    def scoreArgs(argument_list):
        pass


    def setTarget(self, id, revenge):
        pass


def parseArgs(args):
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # need this to ensure -h (for hostname) can be used as an option
    # in optparse before passing the arguments to aiwolfpy
    parser.set_conflict_handler("resolve")

    parser.add_option('-h', action="store", type="string", dest="hostname",
                      help="IP address of the AIWolf server", default=None)
    parser.add_option('-p', action="store", type="int", dest="port",
                      help="Port to connect in the server", default=None)
    parser.add_option('-r', action="store", type="string", dest="port",
                      help="Role request to the server", default=-1)

    (opt, args) = parser.parse_args()
    if opt.hostname == None or opt.port == -1:
        parser.print_help()
        sys.exit()


if __name__ == '__main__':
    parseArgs(sys.argv[1:])
    aiwolfpy.connect_parse(printerAgent("printerAgent"))
