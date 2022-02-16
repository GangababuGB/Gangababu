# #T2-------------------------------------------- Python if-else
# def odd(n):
#     odd_obj = n % 2 == 1
#     return odd_obj
# 
# def even(n):
#     even_obj = n % 2 == 0
#     return even_obj
# 
# if __name__ == '__main__':
#     n = int(input().strip())
#     if odd(n):
#         print('Weird')
#     elif even(n) and n in range(2, 6):
#         print('Not Weird')
#     elif even(n) and n in range(6, 21):
#         print('Weird')
#     else:
#         even(n) and n > 20
#         print('Not Weird')
# #-------------------------------------------------- List comprehension
# if __name__ == '__main__':
#     x = int(input())
#     y = int(input())
#     z = int(input())
#     n = int(input())
#     ans = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n]
#     print(ans)
# #---------------------------------------------------------------------------

# import numpy as np
# if __name__ == '__main__':
#     n = int(input())
#     arr = map(int, input().split())
#     cond = (sorted(list(set(arr)))[-2])
#     print(cond)
# #---------------------------------------------------------------------------
# if __name__ == '__main__':
#     op = []
#     for _ in range(int(input())):
#         name = input()
#         score = float(input())
#         # L1 = [score, name]
#         L1 = [name, score]
#         op.append(L1)
#
# d = sorted(op)
# print(d)
# #---------------------------------------------------------------------------
# persons_dict = {x[0]: x[1:] for x in op}
# print(persons_dict)
# temp = min(persons_dict.values())
# res = [key for key in persons_dict if persons_dict[key] == temp]
#
# # printing result
# # print("Keys with minimum values are : " + str(res))
# temp1=sorted(persons_dict.values())[2]
# res1 = [key for key in persons_dict if persons_dict[key] == temp1]
# # print("Keys with minimum values are : " + str(res1))
#
# for _ in res1:
#     print(_)

# #---------------------------------------------------------------------------
# if __name__ == '__main__':
#     Result = []
#     scorelist = []
#     for _ in range(int(input())):
#         name = input()
#         score = float(input())
#         Result+=[[name, score]]
#         scorelist+=[score]
#     b=sorted(list(set(scorelist)))[1]
#
#     for a, c in sorted(Result):
#         if c==b:
#             print(a)
# #---------------------------------------------------------------------------
# if __name__ == '__main__':
#     n = int(input())
#     avg = 0
#     student_marks = {}
#     for _ in range(n):
#         name, *line = input().split()
#         scores = list(map(float, line))
#         student_marks[name] = scores
#     query_name = input()
#     print("{:.2f}".format(sum(student_marks.get(query_name))/3))
# #----------------------------------------------------------------------------------

# import os
# import pandas as pd
# import re
# path = input(r'path1:')
# directory = os.listdir(path)
# df = pd.DataFrame(directory, columns=['source'])
# f_name = df.source.str.split('.').str[0]
# f_type = df.source.str.split('.').str[1]
# df2 = pd.concat([f_name, f_type], axis=1, join='inner')
# df2.columns = ['Name', 'Type']
#
# path = input(r'path2:')
# directory = os.listdir(path)
# df = pd.DataFrame(directory, columns=['source'])
# f_name = df.source.str.split('.').str[0]
# f_type = df.source.str.split('.').str[1]
# df3 = pd.concat([f_name, f_type], axis=1, join='inner')
# df3.columns = ['Name', 'Type']
#
# dd = pd.merge(df2, df3, how='inner')
# print(dd)
# # F:\Elakkiya\2022\JANUARY\04.01.2022\after-covid\original data
# # F:\Elakkiya\2022\JANUARY\04.01.2022\after-covid\original data2

# # 7 List Operations ------------------------------
# if __name__ == '__main__':
#     N = int(input())
#     L=[]
#     for i in range(0, N):
#         cmd=input().split()
#         if cmd[0] == "insert":
#             L.insert(int(cmd[1]), int(cmd[2]))
#         elif cmd[0] == "print":
#             print(L)
#         elif cmd[0] == "remove":
#             L.remove(int(cmd[1]))
#         elif cmd[0] == "append":
#             L.append(int(cmd[1]))
#         elif cmd[0] == "sort":
#             L.sort()
#         elif cmd[0] == "pop":
#             L.pop()
#         else:
#             L.reverse()

# #-----------------------------------

# if __name__ == '__main__':
#     # n = int(input())
#     integer_list = list(map(int, input().split()))
#     t = tuple(integer_list)
#     print(-hash(t))

# #-------------------
# import numpy as np
# x2 = np.array([[3, 7, 5, 5],
#       [0, 1, 5, 9],
#       [3, 0, 5, 0]])
# x2[2, -4] = 91
# print(x2[2, -4])

# #------------------- Mutation
# def mutate_string(string, position, character):
#     string = string[:position]+character+string[position+1:]
#     return string
#
# if __name__ == '__main__':
#     s = input()
#     i, c = input().split()
#     s_new = mutate_string(s, int(i), c)
#     print(s_new)
# #-------------------------------------------------- Find a string
# def count_substring(string, sub_string):
#     k=0
#     for i in range(len(string)):
#         if string[i:i + (len(sub_string))] == sub_string:
#             k+=1
#     return k
#
# if __name__ == '__main__':
#     string = input().strip()
#     sub_string = input().strip()
#
#     count = count_substring(string, sub_string)
#     print(count)
# #------------------------------------------------------------
# if __name__ == '__main__':
#     s = input()
#     print(any(c.isalnum() for c in s))
#     print(any(c.isalpha() for c in s))
#     print(any(c.isdigit() for c in s))
#     print(any(c.islower() for c in s))
#     print(any(c.isupper() for c in s))
# #-------------------------------------------------------- pattern
#Replace all ______ with rjust, ljust or center.

# thickness = int(input()) #This must be an odd number
# c = 'H'

# #Top Cone
# for i in range(thickness):
#     print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#
# #Top Pillars
# for i in range(thickness+1):
#     print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#
# #Middle Belt
# for i in range((thickness+1)//2):
#     print((c*thickness*5).center(thickness*6))
#
# #Bottom Pillars
# for i in range(thickness+1):
#     print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#
# #Bottom Cone
# for i in range(thickness):
#     print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# #---------------------------------------------- Textwrap

# import textwrap
#
# def wrap(string, max_width):
#     return textwrap.TextWrapper(width=max_width).fill(text=string)
#
# if __name__ == '__main__':
#     string, max_width = input(), int(input())
#     result = wrap(string, max_width)
#     print(result)

# #------------------------
# x, y = map(int, input().split())
# items = list(range(1, x+1, 2))
# items = items+items[::-1][1:]
# for i in items:
#     text= "WELCOME" if i == x else '.|.'*i
#     print(text.center(y, '-'))
# #----------------------------------------- #########
# n, m = map(int, input().split())
# pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
# print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))

# #---------------------------------------- String Formatting
# def print_formatted(number):
#     l1 = len(bin(number)[2:])
#
#     for i in range(1, number + 1):
#         print(str(i).rjust(l1, ' '), end=" ")
#         print(oct(i)[2:].rjust(l1, ' '), end=" ")
#         print(((hex(i)[2:]).upper()).rjust(l1, ' '), end=" ")
#         print(bin(i)[2:].rjust(l1, ' '), end=" ")
#         print("")
#
# if __name__ == '__main__':
#     n = int(input())
#     print_formatted(n)
# #----------------------------------------------- Alphabet Rangoli
# def print_rangoli(size):
#     import string
#     design = string.ascii_lowercase
#
#     L = []
#     for i in range(n):
#         s = "-".join(design[i:n])
#         L.append((s[::-1] + s[1:]).center(4 * n - 3, "-"))
#     print('\n'.join(L[:0:-1] + L))
#
# if __name__ == '__main__':
#     n = int(input())
#     print_rangoli(n)
# #------------------------------------------- Capitalize
# import os,sys,math,re
#
# def solve(s):
#     # s = s.split(' ')
#     # l = []
#     # for i in s:
#     #     i = i.capitalize()
#     #     l.append(i)
#     # return l
#     words = s.split(" ")
#     capitalized_words = [w.capitalize() for w in words]
#     return " ".join(capitalized_words)
#
#
# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')
#     s = input()
#     result = solve(s)
#     fptr.write(result + '\n')
#     fptr.close()
# #-------------------------------------------minion game
# def minion_game(s):
#     s1=0
#     s2=0
#     vow=list('aeiou'.upper())
#     # print(vow)
#     for i in range(len(s)):
#         if s[i] not in vow:
#             s1=s1+(len(s)-i)
#             # print('s1', s1, i, s[i])
#         else:
#             s2=s2+(len(s)-i)
#             # print('s2', s2, i, s[i])
#     if s1>s2:
#         print("Stuart", s1)
#     elif s2>s1:
#         print("Kevin", s2)
#     else:
#         print("Draw")
#
# if __name__ == '__main__':
#     s = input()
#     minion_game(s)

# #----------------------------------------------

