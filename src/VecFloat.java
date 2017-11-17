class VecFloat {
    final float[] vals;

    VecFloat(float ... vals){
        this.vals = new float[vals.length];
        for(int i = 0 ; i < vals.length ; i++){
            this.vals[i] = vals[i];
        }
    }
}
