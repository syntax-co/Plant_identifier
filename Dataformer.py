import os,ast
import numpy as np

class dataformer:
    def __init__(self):
        self.leaf_path='leaves'
        self.index_path='indexed_images.txt'
    
    def get_index_file(self):
        file=open(self.index_path,'r')
        info=file.read()
        file.close()
        info=ast.literal_eval(info)
        return info
        
    def leaf_data(self):
        return self.get_index_file()

    def overwrite_index(self,new_form):
        file=open(self.index_path,'w')
        file.write(str(new_form))
        file.close()

    def next_dires(self,path):
        info= next(os.walk(path))
        return info
    
    def reform_image(self,image):
        image=np.moveaxis(image,-1,0)
        return image
    
        
    def form_data(self):
        new_form={"diseased":[],"healthy":[]}
        
        if not os.path.exists(self.index_path):
            file=open(self.index_path,'w')
            file.write(str(new_form))
            file.close()
        
        
        info=self.next_dires(self.leaf_path)
        
        for dire in info[1]:
            dpath=os.path.join(info[0],dire)
            name=dire.split(' (')[0]
            for root,dires,files in os.walk(dpath):
                
                for file in files:
                    fpath=os.path.join(root,file)
                    status=root.split('\\')[2]
                    
                    new_form[status]+=[[name,status,fpath]]
                    
        self.overwrite_index(new_form)
            
    
                    
            
            
            
        
        
        
        
        
                
                
            
            
            
        
        
        
        
        
        
        




if __name__=='__main__':
    
    former=dataformer()
    
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

