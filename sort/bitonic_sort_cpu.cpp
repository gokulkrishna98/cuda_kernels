#include <bits/stdc++.h>
using namespace std;

void bitonic_sort(vector<int> &arr){
    int n = arr.size();
    for(int len = 2; len <= n; len *= 2){
        for(int j=len/2; j>0; j /= 2){
            for(int i=0; i<n; i++){
                int l = i ^ j;
                if(l > i){
                    // if dir = 0, then it should be ascending, else dir should be descending.
                    int dir = i & len;
                    if(((dir == 0) && arr[i] > arr[l]) || ((dir != 0) && arr[i] < arr[l])){
                        swap(arr[i], arr[l]);
                    }
                }
            }
        }
    }
}

void print_array(vector<int> &arr){
    for(auto it: arr){
        printf("%d ", it);
    }
    printf("\n");
}

int main(){
    int n = (1 << 3);
    printf("size: %d\n", 16);
    std::srand(unsigned(std::time(nullptr)));
    vector<int> arr(n);
    std::generate(arr.begin(), arr.end(), std::rand);

    print_array(arr);
    bitonic_sort(arr);
    print_array(arr);

    return 0;
}