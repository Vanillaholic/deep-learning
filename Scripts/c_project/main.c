#include "test.h"
#include <stdio.h>
int main(){
    int n1, n2;
    printf("请输入两个数字，中间用空格分开：");
    scanf("%d\n%d", &n1, &n2);
    test(n1, n2);

    return 0;
}