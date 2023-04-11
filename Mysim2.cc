#include "ns3/vector.h"
#include "ns3/string.h"
#include "ns3/socket.h"
#include "ns3/double.h"
#include "ns3/config.h"
#include "ns3/log.h"
#include "ns3/command-line.h"
#include "ns3/mobility-model.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/position-allocator.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-interface-container.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
 
#include "ns3/ocb-wifi-mac.h"
#include "ns3/wifi-80211p-helper.h"
#include "ns3/wave-mac-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"

#include "ns3/netanim-module.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ns3-ai-module.h"
 
using namespace ns3;
using namespace std;
 
NS_LOG_COMPONENT_DEFINE ("WifiSimpleOcb");

//*GLOBAL VALUES
map<Ipv4Address, uint32_t> nodeId_Ip_map;
uint8_t m_server = 1;
uint32_t numberOfRounds = 300;
double participationRate = 1.0;
uint32_t transmittingNode = -1;

int dropped_list[] = {-1}; //*-1 is used to make array nonempty
int packet_counter = 0;

//*Federated Learning
struct Env {
  int clientId;
  bool isRoundFinished;
  bool packetDroppedFlag;
  int clientToBeDiscarded;
  bool clientUpdateFlag;
} Packed;

struct Act {
  double accuracy;
  double server_accuracy;
  double clientUpdateDuration;
} Packed;

class FL : public Ns3AIRL<Env, Act> {
public:
  FL(uint16_t id);
  double GetClientReturnAccuracy(int clientId);
  double GetClientUpdateDuration(void);
  void SetIsRoundFinished(bool input);
  double SetCommRoundGetAccuracy(void);
  void DiscardClient(int clientId);
  void SetClientDiscardFlag(bool flag);
  void SetClientUpdateFlag(bool flag);
};

FL::FL(uint16_t id) : Ns3AIRL<Env, Act>(id) {
  SetCond(2, 0);
}
Ptr<FL> createFlPtr(uint16_t id) {
  Ptr<FL> flPtr = Create<FL>(id);
  return flPtr;
}

double FL::GetClientReturnAccuracy(int givenClientId) {
  auto env = EnvSetterCond();
  env->clientUpdateFlag = true;
  env->clientId = givenClientId-1;
  SetCompleted();

  auto act = ActionGetterCond();
  double ret = act->accuracy;
  GetCompleted();
  return ret;
}

double FL::GetClientUpdateDuration() {
  auto act = ActionGetterCond();
  double ret = act->clientUpdateDuration;
  GetCompleted();
  return ret;
}

void FL::SetIsRoundFinished(bool input) {
  auto env = EnvSetterCond();
  env->isRoundFinished = input;

  //env->clientId = -1; // to prevent duplicate client updates.

  SetCompleted();
}

double FL::SetCommRoundGetAccuracy() {
  auto env = EnvSetterCond();
  env->isRoundFinished = true;

  //env->clientId = -1; // to prevent duplicate client updates.

  SetCompleted();
  auto act = ActionGetterCond();
  double ret = act->server_accuracy;
  GetCompleted();

  return ret;
}

void FL::DiscardClient(int clientId)
{
  auto env = EnvSetterCond();
  env->clientToBeDiscarded = clientId;
  env->packetDroppedFlag = true;
  SetCompleted();
}

void FL::SetClientDiscardFlag(bool flag)
{
  auto env = EnvSetterCond();
  env->packetDroppedFlag = flag;
  SetCompleted();
}

void FL::SetClientUpdateFlag(bool flag)
{
  auto env = EnvSetterCond();
  env->clientUpdateFlag = flag;
  SetCompleted();
}
//*End of Federated Learning Module


//*Client Module
class FlClientApp : public Application {
  public:
    FlClientApp();
    virtual ~FlClientApp();
    void ClientOnOff(bool isEnabled);

    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate, Ipv4Address ipAddress,Ptr<FL> fl);

    uint32_t m_nPackets;
    double m_packetIntervalMultiplier;
    uint32_t m_successfullySentPackets;
    bool m_isParticipating;
    Ptr<Socket> getSocket(void);

  private:
    virtual void StartApplication(void);
    virtual void StopApplication(void);

    void SendPacket(void);

    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_packetSize;
    DataRate m_dataRate;

    bool m_running;
    uint32_t m_packetsSent;
    Ipv4Address m_ipAddress;
    EventId m_sendEvent;

    Ptr<FL> mm_fl;
};

FlClientApp::FlClientApp() {
  m_successfullySentPackets = 0;
  m_isParticipating = false;
  m_nPackets = 0;
  m_packetIntervalMultiplier = 1;
  m_socket = 0;
  m_packetSize = 0;
  m_dataRate = 0;
  m_running = false;
  m_packetsSent = 0;
}

FlClientApp::~FlClientApp() {
  m_socket = 0;
}

Ptr<Socket> FlClientApp::getSocket(void) {
  return m_socket;
}

void FlClientApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate, Ipv4Address ipAddress, Ptr<FL> fl) {
  m_socket = socket;
  m_peer = address;
  m_packetSize = packetSize;
  m_nPackets = nPackets;
  m_dataRate = dataRate;
  m_ipAddress = ipAddress;
  mm_fl = fl;
}

void FlClientApp::StartApplication(void) {
  m_running = true;
  m_packetsSent = 0;
  m_socket->Bind();
  m_socket->Connect(m_peer);

  //NS_LOG_UNCOND("Client app has started! Ip is:"<<m_ipAddress<<"\tDestination:"<<m_peer);
}

void FlClientApp::ClientOnOff(bool isEnabled) {
  if (isEnabled) {
    NS_LOG_UNCOND("Attempt to sent packet to:"<<m_peer<<"\tsource address is:"<<m_ipAddress);
    SendPacket();
  }
}

void FlClientApp::StopApplication(void) {
  m_running = false;

  if (m_sendEvent.IsRunning()) {
    Simulator::Cancel(m_sendEvent);
  }

  if (m_socket) {
    m_socket->Close();
  }
}

void FlClientApp::SendPacket (void) {
  
  Ptr<Packet> packet = Create<Packet>(m_packetSize);
  Ipv4Header ipv4Header;
  ipv4Header.SetSource(m_ipAddress);
  packet->AddHeader(ipv4Header);
  
  int result = m_socket->Send (packet);
  if (result < 0) {
    //NS_LOG_UNCOND("Error is occured while sending. Error no: " << m_socket->GetErrno());
    Time tNext_error(Seconds(m_packetIntervalMultiplier * m_packetSize * 8 / static_cast<double>(m_dataRate.GetBitRate())));
    m_sendEvent = Simulator::Schedule(tNext_error, &FlClientApp::SendPacket, this);
  }
  else {
    NS_LOG_UNCOND("Sent! "<<result);
  }
  //*Check if all packet are sent
  if (++m_packetsSent < m_nPackets) {
    if(m_running) {
      //*New Packet To Sent Simulator::Now()
      Time tNext(Seconds(m_packetIntervalMultiplier * m_packetSize * 8 / static_cast<double>(m_dataRate.GetBitRate())));
      //m_sendEvent = Simulator::Schedule(Simulator::Now()+Seconds(1.0), &FlClientApp::SendPacket, this);
      m_sendEvent = Simulator::Schedule(tNext, &FlClientApp::SendPacket, this);
    }
  }
  else {
    //*Reset the client and stop sending
    result = m_socket->Send (packet);
    m_packetsSent = 0; 
  }
}
//*End of Client Module


//*Server Module
class FlServerApp : public Application {
  public:
    FlServerApp();
    void Setup(vector<Ptr<FlClientApp>> clientAppList, uint32_t numberOfRounds, double participationRate, uint32_t numOfClients,
               bool isAdaptive, uint32_t participationLevel, Ptr<FL> fl);
    static TypeId GetTypeId(void);
    void StartNextRound(void);
    vector<double> CalculateRoundSuccessPercentage(void);
    void ResetClientParticipationAndSuccess(void);
    void AdjustParticipationRate();

    TracedValue<int32_t> roundChecker;
    vector<Ptr<FlClientApp>> m_clientAppList;
    void GradientPacketReceived(void);

    Ptr<FL> m_fl;
  
  private:
    virtual void StartApplication(void);
    virtual void StopApplication(void);
    vector<uint32_t> GetRandomlySelectedClients(uint32_t clientNum);

    uint32_t m_roundNumber;
    uint32_t m_numberOfRounds;
    uint32_t m_numOfClients;
    bool m_running;
    double m_participationRate;

    Time m_RoundFinishTimeInSec = Seconds(0.);
    map<uint32_t, double> m_roundSuccesPercentageMap;
    uint32_t m_roundLimit;
    bool m_adaptiveParticipationEnabled;
    uint32_t m_participationLevel;

};

/*
static void StartNextRoundCaller(Ptr<FlServerApp> serverApp){
  serverApp->StartNextRound();
}
*/

FlServerApp::FlServerApp() {
  m_roundNumber = 1;
  m_numberOfRounds = 0;
  m_running = false;
  m_roundLimit = 30;
}

/* static */
TypeId FlServerApp::GetTypeId(void) {
  static TypeId tid = TypeId("FlServerApp")
                          .SetParent<Application>()
                          .SetGroupName("Tutorial")
                          .AddConstructor<FlServerApp>()
                          .AddTraceSource("RoundCheckerTrace",
                                          "An int value to trace communication rounds.",
                                          MakeTraceSourceAccessor(&FlServerApp::roundChecker),
                                          "ns3::TracedValueCallback::RoundTrace");
  return tid;
}

void FlServerApp::ResetClientParticipationAndSuccess() {
  for(Ptr<FlClientApp> client : m_clientAppList) 
  {
    client->m_isParticipating = false;
    client->m_successfullySentPackets = 0;
  }
}

vector<double> FlServerApp::CalculateRoundSuccessPercentage() {
  vector<double> rndSucc_total_succsfl;
  double roundSuccessPercentage = 0;
  double participatingTotal = 0;
  double successTotal = 0;
  for(size_t i = 0; i < m_clientAppList.size(); i++) {
    Ptr<FlClientApp> client = m_clientAppList[i];
    if (client->m_isParticipating) {
      participatingTotal++;
    }
    //NS_LOG_UNCOND("Successfuly sent packets total: " << client->m_successfullySentPackets);
    //NS_LOG_UNCOND("m_nPackets is : " << client->m_nPackets);
    if (client->m_successfullySentPackets == client->m_nPackets) {
      successTotal++;
    }
  }
  NS_LOG_UNCOND("Calculated successTotal: " << successTotal);
  NS_LOG_UNCOND("Calculated participatingTotal: " << participatingTotal);
  roundSuccessPercentage = successTotal / participatingTotal;
  NS_LOG_UNCOND("Calculated roundSuccessPercentage: " << roundSuccessPercentage);
  rndSucc_total_succsfl.push_back(roundSuccessPercentage);
  rndSucc_total_succsfl.push_back(participatingTotal);
  rndSucc_total_succsfl.push_back(successTotal);
  return rndSucc_total_succsfl;
}

void FlServerApp::AdjustParticipationRate() {
  if (m_roundNumber == 1) {
    m_participationRate = m_participationRate / 10 * m_participationLevel;
    NS_LOG_UNCOND("New m_participationRate: " << m_participationRate);
  }
}

void FlServerApp::Setup(vector<Ptr<FlClientApp>> clientAppList, uint32_t numberOfRounds, double participationRate, uint32_t numOfClients,
               bool isAdaptive, uint32_t participationLevel, Ptr<FL> fl) {
  NS_LOG_UNCOND("Server app has initialized!");
  m_clientAppList = clientAppList;
  m_numberOfRounds = numberOfRounds;
  m_participationRate = participationRate;
  m_numOfClients = numOfClients;
  m_adaptiveParticipationEnabled = isAdaptive;
  m_participationLevel = participationLevel;
  m_fl = fl;
}

static void ScheduleGradientUpdate(uint32_t clientId, vector<Ptr<FlClientApp>> m_clientAppList) {
  m_clientAppList[clientId]->ClientOnOff(true);
}

static void RoundCheckerCall(Ptr<FlServerApp> serverApp){
  serverApp->roundChecker++;
}

vector<uint32_t> FlServerApp::GetRandomlySelectedClients(uint32_t clientNum) {
  vector<uint32_t> clientList;
  double min = 0.0;
  vector<uint32_t>::iterator it;
  Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable>();
  x->SetAttribute("Min", DoubleValue(min));
  //*client number minus one, last client for bg traffic
  x->SetAttribute("Max", DoubleValue(m_numOfClients-1));
  //*last leaf for bg traffic, thus leftcount - 1
  for (size_t i = 0; i < clientNum; i++) {
    uint32_t randomNumber = x->GetInteger();
    it = std::find (clientList.begin(), clientList.end(), randomNumber);
    while (it != clientList.end()) {
      NS_LOG_UNCOND(randomNumber << " is repeated!");
      randomNumber = x->GetInteger();
      it = std::find (clientList.begin(), clientList.end(), randomNumber);
      NS_LOG_UNCOND("Now trying..." << randomNumber);
    }

    clientList.push_back(randomNumber);
  }
  return clientList; 
}

void FlServerApp::StartNextRound() {
  uint32_t clientNumInRounds = static_cast< float >(m_numOfClients) * static_cast< float >(m_participationRate);
  // for all clients that are going to participate
  Time communicationRoundLimit(Seconds(m_roundLimit));
  bool canIntervalBeFlexible = false;
  uint32_t maxClientForRound = floor(m_roundLimit / 1.58); // 1.58 = tr_delay + queue_delay
  if (maxClientForRound > clientNumInRounds) {
    canIntervalBeFlexible = true;
  }   
  m_RoundFinishTimeInSec = Simulator::Now() + communicationRoundLimit;
  NS_LOG_UNCOND("m_RoundFinishTimeInSec: " << m_RoundFinishTimeInSec.GetSeconds());
  Time delay = Seconds(0.1);
  vector<uint32_t> clientList = GetRandomlySelectedClients(clientNumInRounds);
  for (uint32_t i = 0; i < clientNumInRounds; i++) {
    uint32_t clientIndex = clientList[i];
    
    //*Discard Client
    bool is_drop = std::find(std::begin(dropped_list), std::end(dropped_list), clientIndex) != std::end(dropped_list);
    if (is_drop) {
      NS_LOG_UNCOND("client to be discarded is : " << clientIndex);
      continue;
    }
    
    NS_LOG_UNCOND("clientIndex is : " << clientIndex-1);
    uint32_t pktSize = 936360, fragSize = 63507;
    Time transmission_delay(Seconds(pktSize / static_cast<double>(DataRate("27Mbps").GetBitRate())));
    
    Time queue_delay(MilliSeconds(ceil(pktSize / fragSize) * 100));

    m_clientAppList[clientIndex]->m_isParticipating = true;
    //*apply interval setting below instead of client side
    Simulator::Schedule(delay, &ScheduleGradientUpdate, clientIndex, m_clientAppList);
    
    if (canIntervalBeFlexible) {
      Time maximum_possible_delay(Seconds(static_cast< float >(m_roundLimit) / static_cast< float >(clientNumInRounds)));
      delay = delay + maximum_possible_delay;
      NS_LOG_UNCOND("FlexibleInterval_delay_between_clients: " << delay.GetSeconds());
    }
    else {
      delay = delay + transmission_delay + queue_delay;
      NS_LOG_UNCOND("NOTFlexibleInterval_delay_between_clients: " << delay.GetSeconds());
    }
  }
  if (m_roundNumber < 2) {
    roundChecker++;
  }
  
}

void FlServerApp::StartApplication(void) {
  m_running = true;
  NS_LOG_UNCOND("Server app has started!");
  StartNextRound();
}

void FlServerApp::StopApplication(void) {
  m_running = false;
}
//*End of Server Module


/////////////////////////////////////////////////////////////
void ReceivePacket (Ptr<FlServerApp> serverApp, Ptr<Socket> socket) {
  Ptr<Packet> packet = socket->Recv();
  Ipv4Header ipv4Header;
  map<Ipv4Address, uint32_t>::iterator it;

  packet->RemoveHeader(ipv4Header);

  it = nodeId_Ip_map.find(ipv4Header.GetSource());
  NS_LOG_UNCOND("Received one packet! Its size: " << packet->GetSize()<< "   its id:"<<it->second);
  if (it == nodeId_Ip_map.end()) {
    NS_LOG_UNCOND("Given IP is not a federated learning client, in this case it is background traffic!");
    NS_LOG_UNCOND("Its address:"<<ipv4Header.GetSource());
  }
  else {
    NS_LOG_UNCOND("Given IP and corresponding node ID values : " << it->first << "->" << it->second);
    
    Ptr<FlClientApp> recvdClient = serverApp->m_clientAppList[it->second-1];
    recvdClient->m_successfullySentPackets++;
    NS_LOG_UNCOND("packet counter:"<<++packet_counter<<"\tnum packet that sent:"<<recvdClient->m_successfullySentPackets);
    if (recvdClient->m_successfullySentPackets == recvdClient->m_nPackets) {
      //*if all of the packets that contain gradient update are transmitted successfully 
      NS_LOG_UNCOND("All Packets are sent!");

      serverApp->m_fl->SetIsRoundFinished(false);
      double client_accuracy = serverApp->m_fl->GetClientReturnAccuracy(it->second);
      NS_LOG_UNCOND("Client training is completed\tid:"<<it->second<<"\tAccuracy:"<<client_accuracy<<"\n");
      serverApp->m_fl->SetClientUpdateFlag(false);
    }
  }
  NS_LOG_UNCOND("***************************************");
}

void FlServerApp::GradientPacketReceived() {
  Time delay = Seconds(1.);
  // check the time, if it is up, then finish the communication round.
  if (Simulator::Now() > m_RoundFinishTimeInSec) {
    /* Finish the round and start new round. */
    NS_LOG_UNCOND(m_roundNumber << ". round is finished at: " << Simulator::Now().GetSeconds());
    vector<double> roundSucc_total_sucsfl = CalculateRoundSuccessPercentage();
    NS_LOG_UNCOND(Simulator::Now().GetSeconds() << "\t"
                       << m_roundNumber
                       << "\t" << roundSucc_total_sucsfl[0]
                       << "\t" << roundSucc_total_sucsfl[1]
                       << "\t" << roundSucc_total_sucsfl[2]);
    Time last_update_delay(MilliSeconds(10));
    double new_accuracy = m_fl->SetCommRoundGetAccuracy();
    NS_LOG_UNCOND(Simulator::Now().GetSeconds() << "\t"<< "Server Round Accuracy:"<< "\t" << new_accuracy<<"\n");

    //*Stop Simulation 
    m_fl->SetFinish();
    Simulator::Stop();
  }
  else {
    NS_LOG_UNCOND("Scheduling once more: " << Simulator::Now().GetSeconds());
    Simulator::Schedule(delay + MilliSeconds(500), &RoundCheckerCall, this);
  }
}

void RoundTrace(Ptr<FlServerApp> serverApp, int32_t oldValue, int32_t newValue) {
  NS_LOG_UNCOND("Round trace at: " << Simulator::Now().GetSeconds());
  // Simulator::Schedule(Simulator::Now() + Seconds(1), &FlServerApp::GradientPacketReceived);
  serverApp->GradientPacketReceived();
}
 
int main (int argc, char *argv[]) {
  int m_client = 1;

  std::string phyMode ("OfdmRate27MbpsBW10MHz");
  //uint32_t packetSize = 1024; // bytes
  uint32_t packetSize = 63507/2;
  uint32_t numPackets = 5;
  double interval = 1.0; // seconds
  bool verbose = false;
  bool trace = false;
  uint16_t power = 8.0;

  //*From Outside Code Variable 
  int SeedRunNumber = 434;
  bool adaptiveParticipationEnabled = false;
  uint32_t participationLevel = 0;
  uint32_t memory_id = 1294;
  std::string bitRate ("27Mbps");

  //*Command Line Arguments
  CommandLine cmd (__FILE__); 
  cmd.AddValue("phyMode", "Wifi Phy mode", phyMode);
  cmd.AddValue("bitRate", "Wifi bit/data rate", bitRate);
  cmd.AddValue("packetSize", "size of application packet sent", packetSize);
  cmd.AddValue("numPackets", "number of packets generated", numPackets);
  cmd.AddValue("interval", "interval (seconds) between packets", interval);
  cmd.AddValue("verbose", "turn on all WifiNetDevice log components", verbose);
  cmd.AddValue("trace", "turn on all Wifi physical layer pcap tracing", trace);
  cmd.AddValue("adaptiveParticipationEnabled", "a bool argument", adaptiveParticipationEnabled);
  cmd.AddValue("participationLevel", "an uint32_t argument", participationLevel);
  cmd.AddValue("SeedRunNumber", "seed number", SeedRunNumber);
  cmd.AddValue("participationRate", "a double argument", participationRate);
  cmd.AddValue("numberOfRounds", "an uint32_t argument", numberOfRounds);
  cmd.AddValue("m_client", "an uint32_t argument", m_client);

  cmd.Parse (argc, argv);

  //*Print Hyperparameters
  NS_LOG_UNCOND("Hyperparameters:");
  NS_LOG_UNCOND(" - SeedRunNumber:"<< "\t" << SeedRunNumber);
  NS_LOG_UNCOND(" - participationRate:"<< "\t" << participationRate);
  NS_LOG_UNCOND(" - numberOfRounds:"<< "\t" << numberOfRounds);
  NS_LOG_UNCOND(" - m_client:"<< "\t" << m_client);
  NS_LOG_UNCOND(" - memory_id:"<< "\t" << memory_id);
  NS_LOG_UNCOND(" - adaptiveParticipationEnabled:"<< "\t" << adaptiveParticipationEnabled);  
  NS_LOG_UNCOND(" - participationLevel:"<< "\t" << participationLevel);

  //*Create Federateing Learning Object
  Ptr<FL> fl = createFlPtr(memory_id);

  //*Set Seed
  SeedManager::SetRun(SeedRunNumber);
  RngSeedManager::SetSeed(SeedRunNumber);

  //*Convert To Time Object
  Time interPacketInterval = Seconds (interval);
 
  //*Node Creation
  NodeContainer nodes;
  nodes.Create (m_server+m_client);
 
  //*The below set of helpers will help us to put together the wifi NICs we want
  YansWifiChannelHelper channel;
  YansWifiPhyHelper wifiPhy;
  wifiPhy.SetErrorRateModel ("ns3::NistErrorRateModel");

  channel.AddPropagationLoss ("ns3::FriisPropagationLossModel",
                              "Frequency", DoubleValue (5.180e9));
  channel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiPhy.SetChannel (channel.Create ());
  wifiPhy.Set ("TxPowerStart", DoubleValue (power)); // dBm (1.26 mW)
  wifiPhy.Set ("TxPowerEnd", DoubleValue (power));
  //wifiPhy.Set ("Frequency", UintegerValue (5180));
  
  //*ns-3 supports generate a pcap trace
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11);
  NqosWaveMacHelper wifi80211pMac = NqosWaveMacHelper::Default ();
  Wifi80211pHelper wifi80211p = Wifi80211pHelper::Default ();
  if (verbose) {
      wifi80211p.EnableLogComponents ();      // Turn on all Wifi 802.11p logging
  }
  
  wifi80211p.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                      "DataMode",StringValue (phyMode),
                                      "ControlMode",StringValue (phyMode));
                                      
  NetDeviceContainer devices = wifi80211p.Install (wifiPhy, wifi80211pMac, nodes);

  //*Tracing
  if (trace) {
    wifiPhy.EnablePcap ("wave-simple-80211p", devices);
  }
  
  //*Design Mobility
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  uint8_t j = 0, i = 0, row_count=10; double space=3.0;
  for(uint8_t cnt=0;cnt<m_server+m_client;cnt++) {
    positionAlloc->Add (Vector (i*space, j*space, 0.0));
    mobility.SetPositionAllocator (positionAlloc);
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (nodes);
    if (cnt%row_count==0) {j++;i=0;}i++;
  }
 
  //*Install Internet Stacks 
  InternetStackHelper internet;
  internet.Install (nodes);
  
  //*Assigning Ip Addresses
  Ipv4AddressHelper ipv4;
  NS_LOG_INFO ("Assign IP Addresses.");
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);

  //*Filling Id-Address Map
  for(uint8_t i=m_server;i<m_server+m_client;i++) {
    nodeId_Ip_map.insert(pair<Ipv4Address, uint32_t>(interfaces.GetAddress(i),nodes.Get(i)->GetId()));
  }
  /*
  map<Ipv4Address, uint32_t>::iterator itr;
  for(itr=nodeId_Ip_map.begin(); itr!=nodeId_Ip_map.end(); itr++) {
    NS_LOG_UNCOND((*itr).first<<" : "<<(*itr).second);} 
  */
 
  //*Creating Receiver Socket
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> server = Socket::CreateSocket (nodes.Get (0), tid);
  InetSocketAddress local = InetSocketAddress (interfaces.GetAddress(0), 80);
  server->Bind (local);
  //server->SetRecvCallback (MakeCallback (&ReceivePacket));
 
  //*Creating Sender Socket
  vector<Ptr<Socket>> clientSocketList;
  vector<Ptr<FlClientApp>> flClientAppList;
  for(uint8_t i=m_server;i<m_server+m_client;i++) {
    Ptr<Socket> client = Socket::CreateSocket (nodes.Get (i), tid);
    //InetSocketAddress remote = InetSocketAddress (interfaces.GetAddress(0), 80);
    client->SetAllowBroadcast (true);
    clientSocketList.push_back(client);
    //source->Connect (remote);

    Ptr<FlClientApp> flApp = CreateObject<FlClientApp>();
    flApp->Setup(client, local, packetSize, numPackets, DataRate(bitRate), interfaces.GetAddress(i), fl);
    flClientAppList.push_back(flApp);
    //Simulator::ScheduleWithContext (source->GetNode ()->GetId (),Seconds (1.0+i), &SendPacket,source, packetSize, numPackets, interPacketInterval,interfaces.GetAddress(i));
  }
  
  //*Server app:
  Ptr<FlServerApp> serverApp = CreateObject<FlServerApp>();
  nodes.Get (0)->AddApplication(serverApp);
  serverApp->Setup(flClientAppList, numberOfRounds, participationRate, m_client, adaptiveParticipationEnabled, participationLevel, fl);
  serverApp->SetStartTime(Seconds(0));
  //////////////////////////////////////////
  serverApp->TraceConnectWithoutContext("RoundCheckerTrace", MakeBoundCallback(&RoundTrace, serverApp));
  server->SetRecvCallback (MakeBoundCallback (&ReceivePacket,serverApp));

  //*Client app
  for (uint8_t i=0;i<m_client;i++) {
    nodes.Get (i)->AddApplication(flClientAppList[i]);
    flClientAppList[i]->SetStartTime(Seconds(0));
  }

  NS_LOG_UNCOND("Run Simulation.");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  AnimationInterface anim("study.xml");

  
  Simulator::Run ();
  Simulator::Destroy ();
 
  return 0;
}
