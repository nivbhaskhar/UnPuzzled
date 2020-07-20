#!/usr/bin/env python
# coding: utf-8

#PIL
from PIL import Image
from pylab import array

#Math and numpy
import math
import numpy as np

#Random
from random import seed
from random import randint
from random import sample

#Matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

#Gradio
import gradio as gr
from io import BytesIO
import base64

#Torch
import torch
from torchvision import transforms, utils


def get_puzzle_pieces(image_np):
    """Return the shuffled and randomly rotated puzzle pieces 
       given an image and the dim of each square puzzle piece """
    puzzle_square_piece_dim = 100
    #opening the image as a PIL image object
    my_image = Image.fromarray(image_np)
    
    #getting original image length and width
    original_image_length = my_image.size[0]
    original_image_width =  my_image.size[1] 
    
    #Each puzzle piece is a square of dimension puzzle_square_piece_dim
    puzzle_piece_length = puzzle_square_piece_dim
    puzzle_piece_width = puzzle_square_piece_dim
    
    #Resizing the image so that it can be cut up into integer number of square pieces
    rows = original_image_length // puzzle_piece_length
    cols =  original_image_length // puzzle_piece_width
    new_image_length = rows*puzzle_piece_length
    new_image_width = cols*puzzle_piece_width
    no_of_puzzle_pieces = rows*cols     
    my_image = my_image.resize((new_image_length, new_image_width))
    
  

    
    
    #list_of_labels is [(0,0), (0,1), ..(0, c-1), (1,0), ..(1,c-1),.....(r-1,0), ...(r-1,c-1)]
    


    #We create these r*c number of pieces by cropping the image appropriately
    #We go from top to bottom row and in each row, from left to right cols.
    #For each puzzle piece we create, we also randomly rotate it by 0, 90, 180 or 270 degrees
    #We record how much we rotate each piece by as we do this in a list called puzzle_pieces_orientation
    #We store the puzzle pieces as we generate them in a list called puzzle_pieces
    
    
    puzzle_pieces = []
    puzzle_pieces_orientation = []
    list_of_labels = []
    i = 0
    j = 0
    
    while(i < rows):
        while(j < cols):
            list_of_labels.append((i,j))
            random_int = randint(0,3)
            angle_of_rotation = 90*(random_int)
            puzzle_pieces_orientation.append(random_int)
            puzzle_pieces.append(my_image.crop((j*puzzle_piece_width,i*puzzle_piece_length,(j+1)*puzzle_piece_width,(i+1)*puzzle_piece_length)).rotate(angle_of_rotation))
            j += 1
        i += 1
        j = 0
    
    
    #new_labels is a permutation of list_of_labels
    new_labels = sample(list_of_labels, len(list_of_labels))
   
    #puzzle piece with old label (x,y) gets a new label which is new_label[x*cols + y]  = (a,b) say.
    #i.e., in the shuffled image, the puzzle piece is row a and col b is
    #actually the puzzle piece that was in row x and col y in the original image.
    
    
    #new_to_old_label_dict takes as keys the new labels and returns values as old label 
    #along with the angle of rotations of the puzzle pieces under consideration
    
    top_left_piece_new_label= None
    new_to_old_label_dict = {}
    for new_label, old_label  in zip(new_labels, list_of_labels):
        x, y = old_label
        new_to_old_label_dict[new_label] = (x,y, puzzle_pieces_orientation[x*cols+y])
        if x==0 and y==0:
            top_left_piece_new_label = new_label
            top_left_piece_orientation = puzzle_pieces_orientation[x*cols+y]
    
    
    #We store the shuffled pieces as numpy arrays in a list
    #We again scan the shuffled image from top to botton row and in each row, scan from left to right cols
    
    shuffled_puzzle_pieces_np = []
    
    
    i = 0
    j = 0
    while(i < rows):
        while(j < cols):
            x, y, theta = new_to_old_label_dict[(i,j)] 
            shuffled_piece = puzzle_pieces[x*cols+y]
            shuffled_puzzle_pieces_np.append(array(shuffled_piece))
            j += 1
        i += 1
        j = 0

    
    return rows, cols, top_left_piece_new_label, top_left_piece_orientation, new_to_old_label_dict, shuffled_puzzle_pieces_np
    


def display_puzzle(rows, cols,puzzle_square_piece_dim, shuffled_puzzle_pieces_np, new_to_old_dict, plt_return = True):
    
    puzzle_piece_length = puzzle_square_piece_dim
    puzzle_piece_width = puzzle_square_piece_dim
    new_image_length = rows*puzzle_piece_length
    new_image_width = cols*puzzle_piece_width
    
    if new_to_old_dict is None:
        new_to_old_dict = {}
        for i in range(rows):
            for j in range(cols):
                new_to_old_dict[(i,j)]=(i,j, 0)
    
    solved_image = Image.new('RGB', (new_image_length,new_image_width), color="white")
    for pos, piece in enumerate(shuffled_puzzle_pieces_np):
        new_x, new_y = (pos//cols, (pos % cols))
        if (new_x, new_y) in new_to_old_dict:
            old_x, old_y, r = new_to_old_dict[(new_x, new_y)]
            img_of_piece = Image.fromarray(piece, 'RGB').rotate(-90*r)
            solved_image.paste(img_of_piece,(old_y*puzzle_piece_width,old_x*puzzle_piece_length,(old_y+1)*puzzle_piece_width,(old_x+1)*puzzle_piece_length))
           
    if not plt_return:
        return solved_image
        
    #font size + line thickness
    fig = plt.figure(dpi = 100)
    
    #add subplot
    ax=fig.add_subplot(1,1,1)
    
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    
    #Set up ticks
    Interval=puzzle_square_piece_dim
    loc = plticker.MultipleLocator(base=Interval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-',color='white')
    
    #Add the image
    ax.imshow(solved_image)
    
    return plt


def adjacency_dist(juxtaposed_pieces_torchtensor, width):
    #juxtaposed_pieces_torchtensor = batchsize x channel x height x width
    check = width % 2
    assert (check==0), "Model dim is not even"
    right_edges = juxtaposed_pieces_torchtensor[:, :, :, (width//2)-1]
    left_edges = juxtaposed_pieces_torchtensor[:, :, :, (width//2)]
    differences = left_edges-right_edges
    distances = torch.norm(differences, p='fro', dim=(1,2))
    return distances

class AdjacencyClassifier_NoML():
    def __init__(self,model_dim=224):
        self.model_dim=model_dim

    def negative_distance_score(self, x):
        #x dim is 3 x model_dim x mode_dim
        distances = adjacency_dist(x, self.model_dim)
        return -1*distances
    
    def comparison(self,d,threshold):
        ans = 1
        if d<-1*threshold:
            ans=0
        return ans
    
    def predictions(self,x,threshold):
        distances = self.negative_distance_score(x)
        pred = torch.tensor(list(map(lambda y: self.comparison(y,threshold),distances)))
        return pred






def label_to_pos(label, cols):
        return label[0]*cols + label[1]

def pos_to_label(pos, cols):
        return (pos//cols, (pos % cols))


def transform_puzzle_input(piece_1, piece_2, model_dim=224):
        width = model_dim
        height = model_dim
        piece_1 = piece_1.resize((width, height))
        piece_2 = piece_2.resize((width, height))
        juxtaposed = Image.new('RGB', (2*width, height), color=0)
        #juxtaposed.paste(piece_i ,(left_upper_row, left_upper_col,right_lower_row, right_lower_col))
        juxtaposed.paste(piece_1,(0,0,width, height))
        juxtaposed.paste(piece_2,(width,0,2*width, height))
        juxtaposed = juxtaposed.crop((width//2, 0,width//2 + width,height))
        return transforms.ToTensor()(juxtaposed)
    


def left_right_adj_score(P, Q, R, S, model_name, model):   
    #rotate Puzzle piece P by 90R degrees clockwise
    #rotate Puzzle piece Q by 90S degrees clockwise
    #model(P,Q)
    piece_1 = Image.fromarray(P, 'RGB').rotate(-90*R)
    piece_2 = Image.fromarray(Q, 'RGB').rotate(-90*S)
    
    with torch.no_grad():
            juxtaposed_pieces_torchtensor = transform_puzzle_input(piece_1, piece_2)
            new_input_torchtensor = juxtaposed_pieces_torchtensor.unsqueeze(0)
            if model_name=="AdjacencyClassifier_NoML":
                score = model.negative_distance_score(new_input_torchtensor).numpy()
                return score[0]
            elif model_name=="RandomScorer":
                return random.random()
            else:
                score = model(new_input_torchtensor).numpy()
                return score[0,1]
                


def compute_and_memoize_score(information_dict, shuffled_puzzle_pieces_np, P, Q, R, S, model_name, model):
    N = len(shuffled_puzzle_pieces_np)
    if ((P,R) not in information_dict):
        information_dict[(P,R)] = {Q: {}}
    elif (Q not in information_dict[(P,R)]):
        information_dict[(P,R)][Q] = {}
    if S not in information_dict[(P,R)][Q]:
        information_dict[(P,R)][Q][S] = left_right_adj_score(shuffled_puzzle_pieces_np[P], 
                                                             shuffled_puzzle_pieces_np[Q], 
                                                             R, S, model_name, model)
        R_ = (R + 2) % 4
        S_ = (S + 2) % 4
        if ((Q,S_) not in information_dict):
            information_dict[(Q,S_)] = {P: {}}
        elif (P not in information_dict[(Q,S_)]):
            information_dict[(Q,S_)][P] = {}
        assert(R_ not in information_dict[(Q,S_)][P]),"Symmetric entry already computed unexpectedly"
        information_dict[(Q,S_)][P][R_] = information_dict[(P,R)][Q][S]
    return information_dict[(P,R)][Q][S]



class PuzzleBoard:

    def __init__(self, rows, cols, information_dict, top_left_piece_new_label, 
                 top_left_piece_orientation, shuffled_puzzle_pieces_np, model_name, model):
        self.rows = rows
        self.cols = cols
        self.information_dict = information_dict
        self.available_pieces = set(range(len(shuffled_puzzle_pieces_np)))
        self.filled_slots = set()
        self.open_slots = {(0,0)}
        self.state = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.state[(i,j)] = [(None, None), 
                                     (None, None),
                                     (None, None), 
                                     (None, None)]
        self.predicted_new_to_old_dict = {}
        self.top_left_piece_new_label = top_left_piece_new_label
        self.top_left_piece_orientation  = top_left_piece_orientation
        self.match = {0:2, 1:3, 2:0, 3:1}
        self.shuffled_puzzle_pieces_np = shuffled_puzzle_pieces_np
        self.model_name=model_name
        self.model=model

    def show_progress(self, puzzle_square_piece_dim):
        return display_puzzle(self.rows, self.cols,puzzle_square_piece_dim, 
                          self.shuffled_puzzle_pieces_np, self.predicted_new_to_old_dict, False)
        

        
    def fit(self, current_piece_pos, current_rotation, current_open_slot):
        self.open_slots.remove(current_open_slot)
        
        current_piece_new_label = self.pos_to_new_label(current_piece_pos)
        self.predicted_new_to_old_dict[current_piece_new_label] = (*current_open_slot, current_rotation)
        
        for i, nbhr_slot in enumerate(self.neighbour_slots(current_open_slot)):
            if nbhr_slot is not None:
                self.state[nbhr_slot][self.match[i]] = (current_piece_pos, current_rotation)
                if nbhr_slot not in self.filled_slots:
                    self.open_slots.add(nbhr_slot)
                    
        self.available_pieces.remove(current_piece_pos)
        self.filled_slots.add(current_open_slot)

        
       
            
        
    def fit_top_left_corner(self):
        top_left_piece_pos = self.new_label_to_pos(self.top_left_piece_new_label)
        self.fit(top_left_piece_pos, self.top_left_piece_orientation, (0,0))
        
        
    def find_best_fit(self):
        candidate_open_slot = None
        candidate_pos = None
        candidate_rotation = None
        best_score = -math.inf
        for open_slot in self.open_slots:
            for current_piece_pos in self.available_pieces:
                for current_rotation in [0,1,2,3]:
                    sum_of_scores = 0
                    no_of_nbhrs = 0
                    for i in range(4):
                        if self.state[open_slot][i][0] is not None:
                            nbhr = self.state[open_slot][i][0]
                            nbhr_rotation = self.state[open_slot][i][1]
                            NR_ = (nbhr_rotation-i) % 4
                            R_ = (current_rotation-i) % 4
                            current_score = compute_and_memoize_score(self.information_dict,
                                                                      self.shuffled_puzzle_pieces_np,
                                                                      nbhr,current_piece_pos,
                                                                      NR_,R_,self.model_name,self.model)
                            sum_of_scores += current_score
                            no_of_nbhrs += 1
                    if no_of_nbhrs > 0:
                        score = sum_of_scores/no_of_nbhrs
                    else: 
                        score = 0

                    if score > best_score:
                        candidate_pos = current_piece_pos
                        candidate_rotation = current_rotation
                        candidate_open_slot = open_slot
                        best_score = score
                        
        return (candidate_pos, candidate_rotation, candidate_open_slot)        
        
    ##Helper methods
    
    def neighbour_slots(self, slot):
        p, q = slot
        nbhr_slots = [None, None, None, None]
        nbhr_slots_candidates = [(p,q-1), (p-1,q), (p,q+1), (p+1,q)]
        for i in range(4):
            a,b = nbhr_slots_candidates[i]
            if a>=0 and a< self.rows and b>=0 and b< self.cols:
                nbhr_slots[i] = nbhr_slots_candidates[i]
        return nbhr_slots
    
        
    def new_label_to_pos(self, new_label):
        return new_label[0]*self.cols + new_label[1]

    def pos_to_new_label(self, pos):
        return (pos//self.cols, (pos % self.cols))
    
    



def solve_puzzle(rows, cols, top_left_piece_new_label, top_left_piece_orientation, new_to_old_label_dict,
                 shuffled_puzzle_pieces_np, puzzle_square_piece_dim, model_name, model) :
    
    solver_steps = []
    N = len(shuffled_puzzle_pieces_np)
    information_dict = {}
    board = PuzzleBoard(rows, cols, information_dict, top_left_piece_new_label, 
                           top_left_piece_orientation,
                           shuffled_puzzle_pieces_np,model_name, model)
    
    board.fit_top_left_corner()
    solver_steps.append(board.show_progress(puzzle_square_piece_dim))
    no_of_slots_left =  rows*cols-1
    
    for counter in range(no_of_slots_left):
        candidate = board.find_best_fit() 
        board.fit(*candidate)
        solver_steps.append(board.show_progress(puzzle_square_piece_dim))
        
    


    correct_position = 0
    correct_position_and_rotation = 0
    for k in new_to_old_label_dict:
        if new_to_old_label_dict[k][:2] == board.predicted_new_to_old_dict[k][:2]:
            correct_position += 1
            if new_to_old_label_dict[k][2] == board.predicted_new_to_old_dict[k][2]:
                correct_position_and_rotation += 1
                
    no_of_pieces = rows*cols
                
    return no_of_pieces, correct_position, correct_position_and_rotation, solver_steps


def generate_and_solve_puzzle(image_for_puzzle):
    
        model_name = 'AdjacencyClassifier_NoML'
        model =  AdjacencyClassifier_NoML()
        puzzle_square_piece_dim=100
        
        #Generate puzzle
        puzzle = get_puzzle_pieces(image_for_puzzle)
        rows, cols, top_left_piece_new_label, top_left_piece_orientation, new_to_old_label_dict, shuffled_puzzle_pieces_np = puzzle
        
        #Shuffled puzzle plot
        shuffled_puzzle_plt = display_puzzle(rows, cols,puzzle_square_piece_dim,
                                             shuffled_puzzle_pieces_np, None) 
        
        #Solver outputs
        outputs = solve_puzzle(rows, cols, top_left_piece_new_label, top_left_piece_orientation, new_to_old_label_dict,
                 shuffled_puzzle_pieces_np, puzzle_square_piece_dim, model_name, model)
        no_of_pieces, correct_position, correct_position_and_rotation, solver_steps = outputs
        eval_results = str(correct_position_and_rotation)+"/" + str(no_of_pieces)
        
        return shuffled_puzzle_plt, solver_steps, eval_results


        
        
        
        


def encode_to_gif(PIL_images):
    with BytesIO() as output_bytes:
        PIL_images[0].save(output_bytes, 'GIF', save_all=True, append_images=PIL_images[1:], optimize=False, duration=1000, loop=0)
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return "data:image/gif;base64," + base64_str


gif_output = gr.outputs.Image(label="Solving..")
gif_output.postprocess = encode_to_gif


gr.Interface(
  generate_and_solve_puzzle, 
  gr.inputs.Image(shape=(400, 400), image_mode="RGB", label="Image to generate puzzle"),
  [gr.outputs.Image(plot=True, label="The puzzle "),
    gif_output,
    gr.outputs.Textbox(label="Pieces in correct position and orientation")
  ], title="Unpuzzler", description="Give us your favourite image. Watch it get *puzzled* and *unpuzzled* by the Unpuzzler!").launch();


