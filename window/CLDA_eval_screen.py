# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 22:07:34 2018

@author: Shotaro Baba
"""

import tkinter as tk
import gc
import main_window



class CLDA_evaluation_screen(object):
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CLDA Evaluation")
        
        
        self.menubar = tk.Menu(self.root)
        self.menubar.add_command(label="Quit", command=None)
        self.menubar.add_command(label="Help", command=None)
        
        self.root.config(menu=self.menubar)
        
        self.start_menu()
        
        self.root.mainloop()
        
    def start_menu(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()
        
        self.bottom_button_frame = tk.Frame(self.root)
        self.bottom_button_frame.pack()
        
        self.result_and_selection_frame = tk.Frame(self.main_frame)
        self.result_and_selection_frame.pack()
        '''
        ##########################################################
        ####Text-based result
        ##########################################################
        '''
        #Showing the result of the calculation in the text field
        self.result_frame = tk.Frame(self.result_and_selection_frame)
        self.result_frame.pack(side = tk.RIGHT, pady = 10, padx = 5)
        
        self.result_screen_label = tk.Label(self.result_frame, text = "CLDA evaluation result")
        self.result_screen_label.pack()
        
        self.result_screen_text = tk.Text(self.result_frame)
        self.result_screen_text.pack(side = tk.LEFT)
        
        self.result_screen_text_scroll = tk.Scrollbar(self.result_frame)
        self.result_screen_text_scroll.pack(side = tk.RIGHT, fill = 'y')
        
        self.result_screen_text['yscrollcommand'] = \
        self.result_screen_text_scroll.set
        self.result_screen_text_scroll['command'] = self.result_screen_text.yview
        
        #Selection menu for generating algorithms
        
        '''
        ##########################################################
        ####CLDA model selection
        ##########################################################
        '''
        self.CLDA_selection_and_preference = tk.Frame(self.result_and_selection_frame, relief = tk.RAISED, borderwidth = 1)
        self.CLDA_selection_and_preference.pack(side = tk.RIGHT, padx = 10)
        
        
        
        self.CLDA_selection_label = tk.Label(self.CLDA_selection_and_preference, text = "CLDA selection")
        self.CLDA_selection_label.grid(row = 0)
        
        #######################################
        #Scrolling section
        #######################################
        self.CLDA_selection_and_preference_list_scroll = tk.Frame(self.CLDA_selection_and_preference)
        self.CLDA_selection_and_preference_list_scroll.grid(row = 1)
        
        self.CLDA_selection_listbox = tk.Listbox(self.CLDA_selection_and_preference_list_scroll)
        self.CLDA_selection_listbox.pack(side = tk.LEFT)
        
        self.CLDA_selection_listbox_scroll = tk.Scrollbar(self.CLDA_selection_and_preference_list_scroll)
        self.CLDA_selection_listbox_scroll.pack(side = tk.RIGHT, fill = 'y')
        
        #########################################
        ##Button section
        #########################################
        
        self.CLDA_word_ranking_input_frame = tk.Frame(self.CLDA_selection_and_preference)
        self.CLDA_word_ranking_input_frame.grid(row = 2)
        
        self.CLDA_selection_word_ranking_label = tk.Label(self.CLDA_word_ranking_input_frame, text = "Enter word\nfor ranking\nfor CLDA:")
        self.CLDA_selection_word_ranking_label.pack(side = tk.LEFT)
        
        self.CLDA_selection_word_ranking_box = tk.Entry(self.CLDA_word_ranking_input_frame)
        self.CLDA_selection_word_ranking_box.pack(side = tk.RIGHT)
        
        
        self.CLDA_selection_word_ranking_button = tk.Button(self.CLDA_selection_and_preference, text = "Word ranking\nevaluation")
        self.CLDA_selection_word_ranking_button.grid(row = 3)
        
        
        self.CLDA_selection_listbox['yscrollcommand'] = \
        self.CLDA_selection_listbox_scroll.set
        self.CLDA_selection_listbox_scroll['command'] = self.CLDA_selection_listbox.yview
        
        '''
        ##########################################################
        ####LDA model selection
        ##########################################################
        '''
        self.LDA_selection_and_preference = tk.Frame(self.result_and_selection_frame, relief = tk.RAISED, borderwidth = 1)
        self.LDA_selection_and_preference.pack(side = tk.RIGHT, padx = 10)
        
        
        
        self.LDA_selection_label = tk.Label(self.LDA_selection_and_preference, text = "LDA selection")
        self.LDA_selection_label.grid(row = 0)
        
        #######################################
        #Scrolling section
        #######################################
        self.LDA_selection_and_preference_list_scroll = tk.Frame(self.LDA_selection_and_preference)
        self.LDA_selection_and_preference_list_scroll.grid(row = 1)
        
        self.LDA_selection_listbox = tk.Listbox(self.LDA_selection_and_preference_list_scroll)
        self.LDA_selection_listbox.pack(side = tk.LEFT)
        
        self.LDA_selection_listbox_scroll = tk.Scrollbar(self.LDA_selection_and_preference_list_scroll)
        self.LDA_selection_listbox_scroll.pack(side = tk.RIGHT, fill = 'y')
        
        #########################################
        ##Button section
        #########################################
        
        self.LDA_word_ranking_input_frame = tk.Frame(self.LDA_selection_and_preference)
        self.LDA_word_ranking_input_frame.grid(row = 2)
        
        self.LDA_selection_word_ranking_label = tk.Label(self.LDA_word_ranking_input_frame, text = "Enter word\nfor ranking\nfor LDA:")
        self.LDA_selection_word_ranking_label.pack(side = tk.LEFT)
        
        self.LDA_selection_word_ranking_box = tk.Entry(self.LDA_word_ranking_input_frame)
        self.LDA_selection_word_ranking_box.pack(side = tk.RIGHT)
        
        
        self.LDA_selection_word_ranking_button = tk.Button(self.LDA_selection_and_preference, text = "Word ranking\nevaluation")
        self.LDA_selection_word_ranking_button.grid(row = 3)
        
        
        self.LDA_selection_listbox['yscrollcommand'] = \
        self.LDA_selection_listbox_scroll.set
        self.LDA_selection_listbox_scroll['command'] = self.LDA_selection_listbox.yview
        
        
        '''
        ##########################################################
        ####CLDA button
        ##########################################################
        '''
        self.change_to_model_creation = tk.Button(self.bottom_button_frame, text = "Return to model creation")
        self.change_to_model_creation.pack()
        self.change_to_model_creation['command'] = self.move_to_model_creation
        
        
        '''
        ##########################################################
        ####LDA button
        ##########################################################
        '''
        
    def move_to_model_creation(self):
        
        #Erase all the display in the directory
        self.root.destroy()
        
        #Delete the object to erase its data
        del self
        gc.collect()
        main_window.main()
        
                
def main():
    
    CLDA_evaluation_screen()
    
if __name__ == "__main__":
    main()