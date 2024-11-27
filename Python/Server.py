import sys, socket,threading,time,re
import heapq
import networkx as nx
import matplotlib.pyplot as plt

from ServerWorker import ServerWorker

class Server:	

    def __init__(self):
        # Initialize the connection data log as an instance variable
        self.connection_data_log = []
        self.points_presence = ['10.0.8.2,10.0.9.2,10.0.11.2']
        self.graph = nx.DiGraph()
        self.ip_to_node = {}
        self.tree = None 
        self.node_to_ips = {}

    
    def main(self):
        threading.Thread(target=self.bootstrapper).start()
        threading.Thread(target=self.rtspSocket).start()  
        server_thread = threading.Thread(target=self.server)
        server_thread.start()



    def server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('0.0.0.0', 24000))
        #print(f"Servidor ouvindo...")

        while True:
            # Escuta por mensagens de qualquer cliente
            message, client_address = server_socket.recvfrom(1024)
            #print(f"Recebido de {client_address}: {message.decode()}")
            client_ip = client_address[0]
            
            # Verifica se a mensagem recebida é "HELLO"
            if message.decode().strip() == "Hello Neighbor":
                # Envia uma resposta de volta para o cliente
                response = f"HELLO {client_address[0]}"
                server_socket.sendto(response.encode(), client_address)
                #print(f"Resposta enviada para {client_address}")
            if message.decode().startswith("REQ:"):
                print(message)
                data = message.decode().split(":", 1)[1]  
                path_str, videofile = data.split("-", 1)  

                print("Original Path String:", path_str)
                print("Video File:", videofile)


                bruh = "Stream Packet" # For Simulation

                response_message2 = f"STREAM:{path_str}-{bruh}"
                server_socket.sendto(response_message2.encode(),client_address)
                print("Sending it back")
                print(f"{client_address}")


                """
                node = self.get_node_by_ip(client_ip)
                node_host = self.get_node_by_ip('10.0.17.10')
                start_node = node_host
                end_node = node
                if start_node in self.graph.nodes and end_node in self.graph.nodes:
                    path, total_time = self.find_shortest_path(start_node, end_node)
                    path2, total_time2 = self.find_shortest_path_with_ips(start_node,end_node)
                    print(f"Path: {path} \n Time: {total_time}")
                    print(f"Path: {path2} \n Time: {total_time2}")

                    content = message.split("REQ: ")[1].strip()
                    videofile, client_ip2 = content.split(",")

                    paths,dads = self.find_all_paths_and_parents(start_node,'10','10')

                       


                    path = path[1:]  
                    path2first = path2[0]
                    next = path2first[1]
                    print(next)
                    path2 = path2[1:]
                    response_message2 = f"STREAM: {client_ip2}, bruh, {path2}"
                    server_socket.sendto(response_message2.encode(), (next, 24000))

                """



    def save_connection_data(self, origin, dest, time):
        # Do not normalize the connection; treat (origin, dest) and (dest, origin) as distinct.
        # Check if an entry with the same origin and dest exists (no sorting).
        for record in self.connection_data_log:
            if record['origin'] == origin and record['dest'] == dest:
                # Update the time if found
                record['time'] = time
                #print(f"Updated time for connection from {origin} to {dest} to {time}")
                return

        # If no existing entry is found, create a new record
        origin_node = self.get_node_by_ip(origin)
        dest_node = self.get_node_by_ip(dest)
        
        connection_info = {
            "origin": origin,
            "dest": dest,
            "time": time,
            "origin_node": origin_node,
            "dest_node": dest_node
        }
        
        # Add the new record to the log
        self.connection_data_log.append(connection_info)
        #print(f"Added new record for connection from {origin} to {dest} with time: {time}")

        self.add_connection_to_graph(origin_node,dest_node,time)

    def add_connection_to_graph(self, origin_node, dest_node, time):
        if not self.graph.has_node(origin_node):
            self.graph.add_node(origin_node)
            #print(f"Added node: {origin_node}")
    
        if not self.graph.has_node(dest_node):
            self.graph.add_node(dest_node)
            #print(f"Added node: {dest_node}")
        
        # Add the edge with the time as the weight
        self.graph.add_edge(origin_node, dest_node, weight=time)
        #print(f"Edge added: {origin_node} -> {dest_node} with time {time}")

    def build_graph_from_connections(self):
        """Builds the graph using the existing connection data."""
        for record in self.connection_data_log:
            origin_node = record["origin_node"]
            dest_node = record["dest_node"]
            time = record["time"]
            self.add_connection_to_graph(origin_node, dest_node, time)

    def find_shortest_path(self, start_node, end_node):
        """Finds the shortest path between two nodes using Dijkstra's algorithm."""
        try:
            # Dijkstra's algorithm to find the shortest path based on time
            path = nx.dijkstra_path(self.graph, source=start_node, target=end_node, weight='weight')
            total_time = nx.dijkstra_path_length(self.graph, source=start_node, target=end_node, weight='weight')
            
            #print(f"Shortest path from {start_node} to {end_node}: {path}")
            #print(f"Total time for the path: {total_time}")
            return path, total_time
        except nx.NetworkXNoPath:
            #print(f"No path found between {start_node} and {end_node}")
            return None, None
        
    def find_shortest_path_with_ips(self, start_node, end_node):
        """
        Finds the shortest path between two nodes and returns it as a list of connections by IP addresses.
        """
        try:
            # Find the shortest path using Dijkstra's algorithm
            node_path = nx.dijkstra_path(self.graph, source=start_node, target=end_node, weight='weight')
            total_time = nx.dijkstra_path_length(self.graph, source=start_node, target=end_node, weight='weight')
            
            # Convert the path of nodes to a path of IP addresses
            ip_path = []
            for i in range(len(node_path) - 1):
                origin_node = node_path[i]
                dest_node = node_path[i + 1]
                
                # Find the connection in the log that matches the nodes
                for record in self.connection_data_log:
                    if record["origin_node"] == origin_node and record["dest_node"] == dest_node:
                        ip_path.append((record["origin"], record["dest"]))
                        break
            
            #print(f"Shortest path (IPs) from {start_node} to {end_node}: {ip_path}")
            #print(f"Total time for the path: {total_time}")
            return ip_path, total_time
        except nx.NetworkXNoPath:
            print(f"No path found between {start_node} and {end_node}")
            return None, None


    def find_all_paths_and_parents(self, start_node, end_node, target_node):
        """
        Finds all paths between two nodes and determines all possible parents of a specific node, showing parents as IPs.
        
        :param start_node: The starting node of the paths.
        :param end_node: The destination node of the paths.
        :param target_node: The node whose parents we want to find.
        :return: A list of all paths, a set of all parents (IPs) for the target node.
        """
        try:
            # Find all simple paths between start_node and end_node
            all_paths = list(nx.all_simple_paths(self.graph, source=start_node, target=end_node))
            
            # Dictionary mapping nodes to their IPs (assumes this exists in your class)
            node_to_ip = {record["origin_node"]: record["origin"] for record in self.connection_data_log}
            
            # Set to store unique parents of the target node as IPs
            parents_ips = set()
            
            for path in all_paths:
                # Check if the target_node exists in the path
                if target_node in path:
                    # Get the index of the target_node in the path
                    index = path.index(target_node)
                    # If target_node is not the first node, add its parent
                    if index > 0:
                        parent_node = path[index - 1]
                        parent_ip = node_to_ip.get(parent_node, "Unknown IP")
                        parents_ips.add(parent_ip)
            
            print(f"All paths from {start_node} to {end_node}: {all_paths}")
            print(f"All parents of {target_node} as IPs: {parents_ips}")
            
            return all_paths, parents_ips
        except nx.NetworkXNoPath:
            print(f"No paths found between {start_node} and {end_node}")
            return None, None


    def build_tree_from_graph(self, root):
        """
        Constructs a tree from the current graph using shortest paths from the specified root.
        Saves the tree to the `self.tree` attribute.
        """
        if root not in self.graph:
            raise ValueError(f"Root node {root} is not in the graph.")
        
        # Compute shortest paths from the root to all nodes
        shortest_paths = nx.single_source_dijkstra_path(self.graph, source=root, weight='weight')
        
        # Create a new graph for the tree
        tree = nx.Graph()

        # Add nodes and edges based on the shortest paths
        for target, path in shortest_paths.items():
            if len(path) > 1:  # Skip the root node itself
                for i in range(len(path) - 1):
                    origin, dest = path[i], path[i + 1]
                    weight = self.graph[origin][dest]['weight']
                    tree.add_edge(origin, dest, weight=weight)
        
        # Save the tree in memory
        self.tree = tree
        #print(f"Tree built from graph with root {root} and saved in memory.")

    
    def display_tree_in_terminal(self, root):
        """
        Displays the tree structure stored in `self.tree` in the terminal as a hierarchy.
        """
        if self.tree is None:
            print("No tree has been built yet. Build a tree first.")
            return

        if root not in self.tree:
            print(f"Root node {root} is not in the tree.")
            return

        def display_node(node, level=0, visited=None):
            if visited is None:
                visited = set()

            visited.add(node)
            print("  " * level + f"- {node}")

            # Traverse neighbors (children) in the tree
            for neighbor in self.tree.neighbors(node):
                if neighbor not in visited:
                    display_node(neighbor, level + 1, visited)

        print("Tree Structure:")
        display_node(root)

    def get_parent(self, node, root):
        """
        Returns the parent of the given node in the tree.
        
        :param node: The node whose parent is to be found.
        :param root: The root of the tree.
        :return: The parent node, or None if the node is the root or not found.
        """
        if self.tree is None:
            print("No tree has been built yet. Build a tree first.")
            return None

        if node not in self.tree:
            print(f"Node {node} is not in the tree.")
            return None

        if node == root:
            print(f"Node {node} is the root and has no parent.")
            return None

        # Find the shortest path from the root to the node
        path = nx.shortest_path(self.tree, source=root, target=node, weight=None)

        # The parent is the second-to-last node in the path
        parent = path[-2]
        return parent

    def rtspSocket(self):
        try:
            SERVER_PORT = int(sys.argv[1])
        except:
            print("[Usage: Server.py Server_port]\n")
            SERVER_PORT = 2000

        # Create a UDP socket
        rtspSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rtspSocket.bind(('', SERVER_PORT))
        print(f"RTSP Server is listening on UDP port {SERVER_PORT}.")

        while True:
            clientInfo = {}
            
            # Receive RTSP request from client
            try:
                data, clientAddress = rtspSocket.recvfrom(40280)  # Buffer size is 1024 bytes
                if data:
                    print(f"Received data from {clientAddress}:\n{data.decode('utf-8')}")
                    clientInfo['rtspSocket'] = (rtspSocket, clientAddress)
                    # Handle the client request in a new ServerWorker
                    ServerWorker(clientInfo).run()
            except Exception as e:
                print(f"Error receiving data: {e}")


	
                
    def bootstrapper(self):
        """Bootstrapper function to read interfaces and neighbors from config.txt."""
        print("Bootstrapper started, reading configuration file.")
        config_file = 'config.txt'
        try:
            SERVER_PORT2 = int(sys.argv[2])
        except:
            print("[Usage: Server.py Server_port]\n")
            SERVER_PORT2 = 2200

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', SERVER_PORT2)) 
        print(f"Esperando mensagem")
        groups = self.load_config(config_file)
        self.load_config2(config_file)
        ip_addresses = self.points_presence[0].split(',')

            
        
        if not groups:
            print("Erro: grupos de IP não carregados corretamente. Verifique o arquivo config.txt.")
            return
        
        while True:
            try:
                sock.settimeout(2)
                self.build_graph_from_connections()
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8').strip()
                
                
                # Extrair o IP do cliente
                client_ip = addr[0]

                if message.startswith("Best IP:"):
                    ip_match = re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', message)
                    if ip_match:
                        extracted_ip = ip_match.group()
                        print(f"Extracted IP: {extracted_ip}")
                    else:
                        print("No IP found")
                    node = self.get_node_by_ip(extracted_ip)
                    node_host = self.get_node_by_ip('10.0.17.10')
                    start_node = node_host
                    end_node = node
                    if start_node in self.graph.nodes and end_node in self.graph.nodes:
                        path, total_time = self.find_shortest_path(start_node, end_node)
                        print(f"Path: {path} \n Time: {total_time}")
                    else:
                        print(f"One of the nodes ({start_node}, {end_node}) does not exist in the graph.")
                    response_message_client= f"{path}"
                    sock.sendto(response_message_client.encode(),addr)



                if message.startswith("TIME :"):
                    parsed_values = message.replace("TIME :", "").strip(",").split(",")

                    ip = parsed_values[0]
                    client_ip2 = parsed_values[1]
                    round_trip_time = float(parsed_values[2])

                    
                    self.save_connection_data(ip, client_ip2,round_trip_time)
                    node_tree = self.get_node_by_ip(client_ip)
                    self.build_tree_from_graph('15')
                    parent = self.get_parent(node_tree,'15')
                    
                    response_tree = f"PARENT: {parent}"
                    sock.sendto(response_tree.encode(),(client_ip,24000))


                if message.startswith("REQ:"):
                    print(message)
                    node = self.get_node_by_ip(client_ip)
                    node_host = self.get_node_by_ip('10.0.17.10')
                    start_node = node_host
                    end_node = node
                    if start_node in self.graph.nodes and end_node in self.graph.nodes:
                        path, total_time = self.find_shortest_path(start_node, end_node)
                        path2, total_time2 = self.find_shortest_path_with_ips(start_node,end_node)
                        print(f"Path: {path} \n Time: {total_time}")
                        print(f"Path: {path2} \n Time: {total_time2}")

                        content = message.split("REQ: ")[1].strip()
                        videofile, client_ip2 = content.split(",")

                        paths,dads = self.find_all_paths_and_parents(start_node,'10','10')

                       


                        path = path[1:]  
                        path2first = path2[0]
                        next = path2first[1]
                        print(next)
                        path2 = path2[1:]
                        response_message2 = f"STREAM: {client_ip2}, bruh, {path2}"
                        sock.sendto(response_message2.encode(), (next, 24000))
                    else:
                        print(f"One of the nodes ({start_node}, {end_node}) does not exist in the graph.")

                    #response_message_client= f"{path}"
                    #sock.sendto(response_message_client.encode(),addr)


                
                if message == ("CLIENT"):
                    print(f"Mensagem recebida de Cliente {client_ip}")
                    extracted_ips = re.findall(r'\((.*?)\)', self.points_presence[0])
                    print(extracted_ips)
                    for ip in extracted_ips:
                        node_ip = self.get_node_by_ip(ip)
                    response_message = f"Pontos de Presenca: {self.points_presence}"
                    sock.sendto(response_message.encode(), addr)

                if message.startswith ("DIED:"):
                    extracted_ip = message.split("DIED :")[1]
                    print(f"Ip {extracted_ip} died")



                    


                # Verificar se a mensagem é do tipo "HELLO
                if message.startswith("HELLO :"):
                    node_id = self.get_node_by_ip(client_ip)
                    #print(f"Mensagem recebida do Nodo {node_id}")
                    
                    # Verifica se o IP está no grupo da esquerda
                    for left_ips, right_ips in groups.items():
                        if client_ip in left_ips:
                            neighbors = []
                            for ip in right_ips:
                                node = self.get_node_by_ip(ip)  # Get the node associated with the IP
                                if node:
                                    neighbors.append(f"{ip} (Node {node})")
                                else:
                                    neighbors.append(f"{ip} (Unknown Node)")

                            response_message = f"Vizinhos: {', '.join(neighbors)}"
                            sock.sendto(response_message.encode(), addr)
                            break  # Send the response once and exit the loop
                    else:
                        print(f"IP {client_ip} não está em nenhum nodo")
                        
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Erro ao receber mensagem: {e}")
            


    def load_config(self, config_file):
        """Lê e processa o arquivo de configuração para carregar grupos de IPs."""
        groups = {}
        
        try:
            with open(config_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    
                    # Verifica se a linha começa com "nX :" e remove esse prefixo
                    if re.match(r"n\d+ :", line):
                        # Remove o prefixo "nX :"
                        line = re.sub(r"^n\d+ :", "", line).strip()

                    # Expressão regular para encontrar os grupos de IPs
                    match = re.match(r"\((.*?)\) - \((.*?)\)", line)
                    if match:
                        # Separar os IPs e preencher os grupos
                        left_ips = match.group(1).split(',')
                        right_ips = match.group(2).split(',')
                        
                        left_ips = [ip.strip() for ip in left_ips]
                        right_ips = [ip.strip() for ip in right_ips]
                        
                        # Adiciona a relação de IPs ao dicionário
                        for left_ip in left_ips:
                            groups[left_ip] = right_ips
                    else:
                        print(f"Erro: Formato inválido na linha: {line}")
        except Exception as e:
            print(f"Erro ao ler o arquivo de configuração: {e}")
        
        return groups
    
    def load_config2(self, config_file):
        """Reads the configuration file and maps IP addresses to nodes."""
        try:
            with open(config_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    
                    # Regex pattern to match lines like n1 :(10.0.7.2,...) - (10.0.0.20,...)
                    match = re.match(r"n(\d+) :\s?\((.*?)\)\s?-\s?\((.*?)\)", line)
                    
                    if match:
                        # Extract node number, left IPs and right IPs
                        node_number = match.group(1)  # Node number e.g., '1', '2', etc.
                        left_ips = match.group(2).split(',')
                        right_ips = match.group(3).split(',')
                        
                        # Remove any extra spaces from IP addresses
                        left_ips = [ip.strip() for ip in left_ips]
                        right_ips = [ip.strip() for ip in right_ips]
                        
                        # Map the IPs to the respective node number in ip_to_node
                        for ip in left_ips:
                            self.ip_to_node[ip] = node_number
                        
                        # Add IPs to the node_to_ips dictionary (node -> IPs)
                        if node_number not in self.node_to_ips:
                            self.node_to_ips[node_number] = []
                        
                        # Add both left and right IPs to the node's IP list
                        self.node_to_ips[node_number].extend(left_ips)
                        
                    else:
                        print(f"Error: Invalid format in line: {line}")
        except Exception as e:
            print(f"Error reading configuration file: {e}")

    def get_node_by_ip(self, ip):
        """Returns the node associated with the provided IP."""
        node = self.ip_to_node.get(ip, None)
        if node is None:
            print(f"Warning: IP {ip} not found in ip_to_node mapping")
        return node

    def get_ips_per_node(self, node):
        """Returns all IP addresses associated with a given node."""
        ips = self.node_to_ips.get(node, None)
        if ips is None:
            print(f"Warning: No IPs found for node {node}")
        return ips



    def construir_grafo(self,config_file, connection_info):
        """
        Construi um grafo representando a rede.

        Args:
            config_file (str): Caminho para o arquivo de configuração.
            connection_info (list): Lista de dicionários com informações de conexão.

        Returns:
            nx.Graph: Grafo representando a rede.
        """

        G = nx.Graph()

        with open(config_file, 'r') as f:
            for line in f:
                node, interfaces = line.strip().split('-')
                interfaces = interfaces.split(',')
                for interface in interfaces:
                    G.add_node(interface)

        # Adicionar as arestas com as latências conhecidas
        for connection in connection_info:
            origin, dest, time = connection['origin'], connection['dest'], connection['time']
            G.add_edge(origin, dest, weight=time)

        return G
    
    def find_best_path(self, start, end):
        # Build the graph from the connection data log
        graph = {}
        for record in self.connection_data_log:
            origin = record['origin']
            dest = record['dest']
            time = record['time']
            
            if origin not in graph:
                graph[origin] = []
            if dest not in graph:
                graph[dest] = []
                
            graph[origin].append((dest, time))
            graph[dest].append((origin, time))  # As it's an undirected graph
        
        # Use Dijkstra's algorithm to find the shortest path
        # Priority queue for Dijkstra's
        pq = [(0, start)]  # (distance, node)
        distances = {start: 0}
        previous_nodes = {start: None}
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # If we reached the destination, reconstruct the path
            if current_node == end:
                path = []
                while previous_nodes[current_node] is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                path.append(start)
                path.reverse()
                return path, distances[end]
            
            # Explore neighbors
            for neighbor, weight in graph.get(current_node, []):
                distance = current_distance + weight
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        # If there's no path from start to end
        return None, float('inf')

if __name__ == "__main__":
    (Server()).main()
    



