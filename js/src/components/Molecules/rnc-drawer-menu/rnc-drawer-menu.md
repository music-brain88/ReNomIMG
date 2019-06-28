# RncDrawerMenu

A Vue implementation of a ReNom Components.  

※ Currently, when 3 pages of ReNomIMG are clicked, mapCurrents setCurrentPage is executed  
　and the value of "$ store.state.current_page" changes.  

※ In the original ReNomIMG, mapActions "init ()" was executed at the time of changing pages, but it was deleted.   
(The process is summarized in app.vue file)  



## Attributes

- menu-obj: Object of menu name and icon class name
- slot: You can use slot.

## UNIT TEST of: RncDrawerMenu

Render:
- Component it renders

Props default:
- Expected existence of 『MenuObj』
- props, for『MenuObj』expecting default as: 『[]』

- Test of method. Expected existence of 『onItemClick』method
- Expected method 『onItemClick』to be a Function
- Expected method 『onItemClick』 with arguments 『train』to return 『"/"』

- Testing method throw error. Expecting the existence of 『onItemClick』
- Expecting error when calling 『onItemClick』 with arguments 『["TEST"]』
